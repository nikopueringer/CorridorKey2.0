#import <Cocoa/Cocoa.h>
#import <dispatch/dispatch.h>

static int launcherStatus = 0;

@interface CorridorKeyLauncher : NSObject <NSApplicationDelegate>
@property(nonatomic, strong) NSTask *task;
@property(nonatomic, strong) NSFileHandle *logFile;
@end

@implementation CorridorKeyLauncher

- (void)showLaunchError:(NSString *)message {
    launcherStatus = 1;
    NSString *logPath = [NSHomeDirectory() stringByAppendingPathComponent:
        @"Library/Logs/CorridorKey2.0/launcher.log"];
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"CorridorKey could not start";
    alert.informativeText = [NSString stringWithFormat:
        @"%@\n\nLauncher log: %@", message, logPath];
    alert.alertStyle = NSAlertStyleCritical;
    [alert addButtonWithTitle:@"Quit"];
    [alert runModal];
    [NSApp terminate:nil];
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    (void)notification;
    NSBundle *bundle = [NSBundle mainBundle];
    NSString *resources = [bundle resourcePath];
    NSString *python = [resources stringByAppendingPathComponent:@"Python/bin/python3"];
    NSString *source = [resources stringByAppendingPathComponent:@"src"];
    NSString *assetRoot = [NSHomeDirectory() stringByAppendingPathComponent:
        @"Library/Application Support/CorridorKey2.0"];
    NSString *logDirectory = [NSHomeDirectory() stringByAppendingPathComponent:
        @"Library/Logs/CorridorKey2.0"];
    NSFileManager *files = [NSFileManager defaultManager];
    NSError *error = nil;
    if (![files createDirectoryAtPath:assetRoot withIntermediateDirectories:YES
                           attributes:nil error:&error] ||
        ![files createDirectoryAtPath:logDirectory withIntermediateDirectories:YES
                           attributes:nil error:&error]) {
        [self showLaunchError:[NSString stringWithFormat:@"Could not create the application folders: %@",
                                                         error.localizedDescription]];
        return;
    }

    NSString *logPath = [logDirectory stringByAppendingPathComponent:@"launcher.log"];
    if (![files fileExistsAtPath:logPath]) {
        [files createFileAtPath:logPath contents:nil attributes:nil];
    }
    self.logFile = [NSFileHandle fileHandleForWritingAtPath:logPath];
    if (self.logFile == nil) {
        [self showLaunchError:@"Could not open the launcher log for writing."];
        return;
    }
    [self.logFile seekToEndOfFile];

    if (![files isExecutableFileAtPath:python] || ![files fileExistsAtPath:source]) {
        [self showLaunchError:@"The app bundle is incomplete. Rebuild or reinstall CorridorKey."];
        return;
    }

    NSMutableDictionary<NSString *, NSString *> *environment =
        [[NSProcessInfo processInfo].environment mutableCopy];
    [environment removeObjectForKey:@"PYTHONHOME"];
    [environment removeObjectForKey:@"PYTHONPATH"];
    NSString *oldPath = environment[@"PATH"] ?: @"/usr/bin:/bin:/usr/sbin:/sbin";
    environment[@"PATH"] = [NSString stringWithFormat:@"/opt/homebrew/bin:/usr/local/bin:%@:%@",
                                                       [python stringByDeletingLastPathComponent], oldPath];
    environment[@"CORRIDORKEY_ASSET_ROOT"] = assetRoot;
    environment[@"CORRIDORKEY_BUNDLED_APP"] = @"1";
    environment[@"PYTHONNOUSERSITE"] = @"1";
    environment[@"PYTHONDONTWRITEBYTECODE"] = @"1";

    self.task = [[NSTask alloc] init];
    self.task.executableURL = [NSURL fileURLWithPath:python];
    NSMutableArray<NSString *> *arguments = [NSMutableArray arrayWithObjects:@"-m", @"gui", nil];
    for (NSString *argument in [[NSProcessInfo processInfo].arguments subarrayWithRange:
                               NSMakeRange(1, [NSProcessInfo processInfo].arguments.count - 1)]) {
        if (![argument hasPrefix:@"-psn_"]) [arguments addObject:argument];
    }
    BOOL selfTest = [arguments containsObject:@"--bundle-self-test"];
    self.task.arguments = arguments;
    self.task.currentDirectoryURL = [NSURL fileURLWithPath:source isDirectory:YES];
    self.task.environment = environment;
    self.task.standardOutput = self.logFile;
    self.task.standardError = self.logFile;
    __weak CorridorKeyLauncher *weakSelf = self;
    self.task.terminationHandler = ^(NSTask *finishedTask) {
        CorridorKeyLauncher *strongSelf = weakSelf;
        if (strongSelf != nil) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [strongSelf.logFile synchronizeFile];
                launcherStatus = finishedTask.terminationStatus;
                if (launcherStatus != 0 && !selfTest) {
                    [strongSelf showLaunchError:@"The Python runtime exited with an error. See the launcher log for details."];
                } else {
                    [NSApp terminate:nil];
                }
            });
        }
    };

    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
    NSError *launchError = nil;
    if (![self.task launchAndReturnError:&launchError]) {
        [self showLaunchError:[NSString stringWithFormat:@"%@", launchError.localizedDescription]];
    }
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    (void)sender;
    return YES;
}

@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSApplication *application = [NSApplication sharedApplication];
        CorridorKeyLauncher *delegate = [[CorridorKeyLauncher alloc] init];
        application.delegate = delegate;
        [application run];
        (void)argc;
        (void)argv;
        return launcherStatus;
    }
}
