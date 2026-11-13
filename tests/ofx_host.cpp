// Minimal host exercises the actual compiled OFX entry points and Unix IPC.
#include <ofxCore.h>
#include <ofxImageEffect.h>
#include <ofxParam.h>
#include <ofxProperty.h>
#include <ofxMessage.h>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// Tests must retain assertions in Release builds.
#undef assert
#define assert(x) do { if (!(x)) { std::cerr << "FAILED " << __LINE__ << ": " << #x << '\n'; std::abort(); } } while (0)
struct Property {
    std::map<std::string, std::vector<int>> i;
    std::map<std::string, std::vector<double>> d;
    std::map<std::string, std::vector<std::string>> s;
    std::map<std::string, std::vector<void*>> p;
};
Property& prop(OfxPropertySetHandle h) { return *reinterpret_cast<Property*>(h); }
template<typename T> int set(std::map<std::string, std::vector<T>>& m, const char* n, int i, T v) {
    auto& a=m[n]; if (a.size() <= size_t(i)) a.resize(i+1); a[i]=v; return kOfxStatOK;
}
template<typename T> int get(std::map<std::string, std::vector<T>>& m, const char* n, int i, T* v) {
    auto it=m.find(n); if(it==m.end() || it->second.size() <= size_t(i)) return kOfxStatErrUnknown;
    *v=it->second[i]; return kOfxStatOK;
}
OfxPropertySuiteV1 ps{}; OfxImageEffectSuiteV1 es{}; OfxParameterSuiteV1 as{}; OfxMessageSuiteV2 ms{};
struct Param { Property property; std::string type; };
struct Clip { Property property, image; std::vector<float> pixels; };
struct Effect { Property property; std::map<std::string, Clip> clips; std::map<std::string, Param> params; };
std::atomic<int> releases{0}, abortRender{0}; std::string lastError;
OfxPropertySetHandle ph(Property& p) { return reinterpret_cast<OfxPropertySetHandle>(&p); }
void init() {
    ps.propSetInt=[](auto h,const char* n,int i,int v){return set(prop(h).i,n,i,v);};
    ps.propSetDouble=[](auto h,const char* n,int i,double v){return set(prop(h).d,n,i,v);};
    ps.propSetString=[](auto h,const char* n,int i,const char* v){
        if (!strcmp(n,kOfxPropPluginDescription)) return kOfxStatErrUnknown; // Resolve 21.2
        return set(prop(h).s,n,i,std::string(v));
    };
    ps.propSetPointer=[](auto h,const char* n,int i,void* v){return set(prop(h).p,n,i,v);};
    ps.propGetInt=[](auto h,const char* n,int i,int* v){return get(prop(h).i,n,i,v);};
    ps.propGetDouble=[](auto h,const char* n,int i,double* v){return get(prop(h).d,n,i,v);};
    ps.propGetPointer=[](auto h,const char* n,int i,void** v){return get(prop(h).p,n,i,v);};
    ps.propGetString=[](auto h,const char* n,int i,char** v){
        auto& m=prop(h).s; auto it=m.find(n); if(it==m.end() || it->second.size()<=size_t(i)) return kOfxStatErrUnknown;
        *v=it->second[i].data(); return kOfxStatOK;
    };
    ps.propGetIntN=[](auto h,const char* n,int count,int* v){ for(int i=0;i<count;i++) {auto s=ps.propGetInt(h,n,i,v+i);if(s)return s;}return kOfxStatOK;};
    es.getPropertySet=[](auto h,OfxPropertySetHandle* p){*p=ph(reinterpret_cast<Effect*>(h)->property);return kOfxStatOK;};
    es.getParamSet=[](auto h,OfxParamSetHandle* p){*p=reinterpret_cast<OfxParamSetHandle>(h);return kOfxStatOK;};
    es.clipDefine=[](auto h,const char* n,OfxPropertySetHandle* p){*p=ph(reinterpret_cast<Effect*>(h)->clips[n].property);return kOfxStatOK;};
    es.clipGetHandle=[](auto h,const char* n,OfxImageClipHandle* c,OfxPropertySetHandle* p){auto& clip=reinterpret_cast<Effect*>(h)->clips[n];*c=reinterpret_cast<OfxImageClipHandle>(&clip);if(p)*p=ph(clip.property);return kOfxStatOK;};
    es.clipGetImage=[](auto c,double,const OfxRectD*,OfxPropertySetHandle* p){*p=ph(reinterpret_cast<Clip*>(c)->image);return kOfxStatOK;};
    es.clipReleaseImage=[](OfxPropertySetHandle){++releases;return kOfxStatOK;};
    es.abort=[](OfxImageEffectHandle){return abortRender.load();};
    as.paramDefine=[](auto h,const char* t,const char* n,OfxPropertySetHandle* p){auto& a=reinterpret_cast<Effect*>(h)->params[n];a.type=t;*p=ph(a.property);return kOfxStatOK;};
    as.paramGetHandle=[](auto h,const char* n,OfxParamHandle* p,OfxPropertySetHandle*){*p=reinterpret_cast<OfxParamHandle>(&reinterpret_cast<Effect*>(h)->params[n]);return kOfxStatOK;};
}
OfxStatus valueAt(OfxParamHandle h,double time,...) {
    auto& p=*reinterpret_cast<Param*>(h);va_list args;va_start(args,time);
    if(p.type==kOfxParamTypeDouble) *va_arg(args,double*)=p.property.d[kOfxParamPropDefault][0];
    else if(p.type==kOfxParamTypeString) *va_arg(args,char**)=p.property.s[kOfxParamPropDefault][0].data();
    else *va_arg(args,int*)=p.property.i[kOfxParamPropDefault][0];
    va_end(args);return kOfxStatOK;
}
OfxStatus message(void*,const char*,const char*,const char* format,...) {
    char text[2048];va_list args;va_start(args,format);vsnprintf(text,sizeof(text),format,args);va_end(args);lastError=text;return kOfxStatOK;
}
const void* suite(OfxPropertySetHandle,const char* name,int) {
    if(!strcmp(name,kOfxPropertySuite))return &ps;
    if(!strcmp(name,kOfxImageEffectSuite))return &es;
    if(!strcmp(name,kOfxParameterSuite))return &as;
    if(!strcmp(name,kOfxMessageSuite))return &ms;
    return nullptr;
}
void image(Clip& c,bool negativeStride=false) {
    // Nonzero origin; padding on every row; optional negative row bytes.
    c.pixels.assign(24, -99.f); c.image.i[kOfxImagePropBounds]={7,11,9,13};
    c.image.i[kOfxImagePropRowBytes]={negativeStride ? -48 : 48};
    c.image.p[kOfxImagePropData]={c.pixels.data()+(negativeStride?12:0)};
    c.image.s[kOfxImageEffectPropPixelDepth]={kOfxBitDepthFloat};
    c.image.s[kOfxImageEffectPropComponents]={kOfxImageComponentRGBA};
    c.image.s[kOfxImageEffectPropPreMultiplication]={kOfxImagePreMultiplied};
}
void exact(int fd,void* p,size_t n,bool write=false) {
#if defined(__APPLE__)
    if(write){int noSigPipe=1;assert(setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&noSigPipe,sizeof(noSigPipe))==0);}
#endif
    while(n){
#if defined(__APPLE__)
        constexpr int sendFlags=0; // SO_NOSIGPIPE is set on listener sockets below.
#else
        constexpr int sendFlags=MSG_NOSIGNAL;
#endif
        ssize_t k=write?send(fd,p,n,sendFlags):recv(fd,p,n,0);assert(k>0);p=static_cast<char*>(p)+k;n-=k;
    }
}
int main(int argc,char** argv) {
    assert(argc==2);init();as.paramGetValueAtTime=valueAt;ms.setPersistentMessage=message;
    ms.clearPersistentMessage=[](void*){lastError.clear();return kOfxStatOK;};
    void* lib=dlopen(argv[1],RTLD_NOW);if(!lib){std::cerr<<dlerror();return 1;}
    auto count=(int(*)())dlsym(lib,"OfxGetNumberOfPlugins");auto getPlugin=(OfxPlugin*(*)(int))dlsym(lib,"OfxGetPlugin");
    assert(count && getPlugin && count()==1 && getPlugin(1)==nullptr);
    auto* plugin=getPlugin(0);OfxHost host{nullptr,suite};plugin->setHost(&host);
    auto call=[&](const char* action,Effect* e=nullptr,Property* in=nullptr,Property* out=nullptr){return plugin->mainEntry(action,e,in?ph(*in):nullptr,out?ph(*out):nullptr);};
    assert(call(kOfxActionLoad)==kOfxStatOK);Effect effect;
    assert(call(kOfxActionDescribe,&effect)==kOfxStatOK);
    assert(call(kOfxImageEffectActionDescribeInContext,&effect)==kOfxStatOK);
    assert(call(kOfxActionCreateInstance,&effect)==kOfxStatOK);
    auto& src=effect.clips["Source"];auto& dst=effect.clips["Output"];image(src,true);image(dst);
    for(int row=0;row<2;row++)for(int x=0;x<2;x++){
        auto p=src.pixels.data()+row*12+x*4;p[0]=0.1f+row*0.1f;p[1]=0.05f;p[2]=0.025f;p[3]=0.5f;
    }
    std::string path="/tmp/ck-ofx-test-"+std::to_string(getpid())+".sock";
    effect.params["socket"].property.s[kOfxParamPropDefault]={path};
    int listener=socket(AF_UNIX,SOCK_STREAM,0);assert(listener>=0);
#if defined(__APPLE__)
    int noSigPipe=1;assert(setsockopt(listener,SOL_SOCKET,SO_NOSIGPIPE,&noSigPipe,sizeof(noSigPipe))==0);
#endif
    sockaddr_un addr{};addr.sun_family=AF_UNIX;strcpy(addr.sun_path,path.c_str());
#if defined(__APPLE__)
    addr.sun_len=static_cast<decltype(addr.sun_len)>(offsetof(sockaddr_un,sun_path)+path.size()+1);
#endif
    unlink(path.c_str()); // Remove a stale endpoint left by an interrupted prior test.
    assert(bind(listener,(sockaddr*)&addr,sizeof(addr))==0);assert(listen(listener,1)==0);
    // Fill the Unix accept backlog before rendering. A busy engine must be
    // retried, not mistaken for a missing/stopped engine. Darwin's Unix
    // listener backlog behavior differs; this retry check is Linux-only.
#if !defined(__APPLE__)
    std::array<int,2> blockers{};
    for (auto& c : blockers) { c=socket(AF_UNIX,SOCK_STREAM,0); assert(connect(c,(sockaddr*)&addr,sizeof(addr))==0); }
#endif
    std::atomic<int> requests{0};
    std::thread server([&]{
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
#if !defined(__APPLE__)
      for (int i=0;i<2;i++) { int c=accept(listener,nullptr,nullptr); assert(c>=0); close(c); }
#endif
      for(int request=0;request<3;request++){
        int c=accept(listener,nullptr,nullptr);assert(c>=0);char header[32];exact(c,header,32);
        assert(!memcmp(header,"CK2REQ01",8));uint32_t bytes;memcpy(&bytes,header+28,4);assert(bytes==64);
        std::vector<float> frame(bytes/4);exact(c,frame.data(),bytes);++requests;
        // Top row was at the beginning of negative-stride storage. Input is unpremultiplied.
        assert(frame[0]>0.19f && frame[0]<0.21f);assert(frame[3]==0.5f);
        char response[16];memcpy(response,"CK2RES01",8);uint32_t status=0;memcpy(response+8,&status,4);memcpy(response+12,&bytes,4);
        if(request==2){uint32_t bad=65;memcpy(response+12,&bad,4);exact(c,response,16,true);close(c);continue;}
        for(size_t i=3;i<frame.size();i+=4)frame[i]=0.25f;
        exact(c,response,16,true);exact(c,frame.data(),bytes,true);close(c);
    }});
    Property in;in.d[kOfxPropTime]={0};in.i[kOfxImageEffectPropRenderWindow]={7,11,9,13};
    assert(call(kOfxImageEffectActionRender,&effect,&in)==kOfxStatOK);assert(requests==1);
#if !defined(__APPLE__)
    for (auto c : blockers) close(c);
#endif
    assert(dst.pixels[0]>0.099f && dst.pixels[0]<0.101f);assert(dst.pixels[3]==0.25f);assert(dst.pixels[8]==-99.f);
    assert(call(kOfxImageEffectActionRender,&effect,&in)==kOfxStatOK);assert(requests==1); // cache hit
    effect.params["output"].property.i[kOfxParamPropDefault]={1};
    assert(call(kOfxImageEffectActionRender,&effect,&in)==kOfxStatOK);assert(requests==1);assert(dst.pixels[0]==0.25f && dst.pixels[3]==1.f);
    effect.params["output"].property.i[kOfxParamPropDefault]={2};
    assert(call(kOfxImageEffectActionRender,&effect,&in)==kOfxStatOK);assert(dst.pixels[0]>0.399f && dst.pixels[0]<0.401f);
    effect.params["despill"].property.d[kOfxParamPropDefault]={0.75};
    assert(call(kOfxImageEffectActionRender,&effect,&in)==kOfxStatOK);assert(requests==2);
    Property changed;changed.s[kOfxPropName]={"refresh"};call(kOfxActionInstanceChanged,&effect,&changed);
    assert(call(kOfxImageEffectActionRender,&effect,&in)==kOfxStatFailed);assert(lastError.find("wrong frame size")!=std::string::npos);
    server.join();close(listener);unlink(path.c_str());
    assert(call(kOfxImageEffectActionRender,&effect,&in)==kOfxStatFailed);assert(lastError.find("Start the OFX engine")!=std::string::npos);
    assert(releases==14); // All acquired images released on success and failure.
    in.s[kOfxImageEffectPropFieldToRender]={kOfxImageFieldLower};
    assert(call(kOfxImageEffectActionRender,&effect,&in)==kOfxStatFailed);
    assert(lastError.find("progressive")!=std::string::npos);assert(releases==14);
    in.s.erase(kOfxImageEffectPropFieldToRender);
    abortRender=1;assert(call(kOfxImageEffectActionRender,&effect,&in)==kOfxStatOK);abortRender=0;
    Property prefs;assert(call(kOfxImageEffectActionGetClipPreferences,&effect,nullptr,&prefs)==kOfxStatOK);
    assert(prefs.s[kOfxImageEffectPropPreMultiplication][0]==kOfxImagePreMultiplied);
    assert(call(kOfxActionDestroyInstance,&effect)==kOfxStatOK);assert(call(kOfxActionUnload)==kOfxStatOK);
    dlclose(lib);std::cout<<"OFX lifecycle, IPC, orientation, alpha, stride, cache, failure and cancellation passed\n";
}
