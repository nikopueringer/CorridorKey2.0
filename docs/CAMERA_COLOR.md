# Camera colour workflow

Camera profiles are optional. The default **Legacy / existing input** retains
the existing sRGB-video and configured image-sequence workflow. Select an
explicit transfer/gamut pair for each clip in the inspector; camera metadata
does not reliably identify every logarithmic recording mode.

## Processing contract

For an explicit camera profile, the source is decoded to normalized camera RGB
code values at 16-bit precision for video, or at the original image precision
for frame folders. An analytic transfer function converts those values to
camera-linear RGB. An RGB matrix then converts the primaries to linear Rec.709.
This is the model's working gamut, not a display Rec.709 rendering transform.
The existing signed-asinh model encoding preserves negative values and HDR
highlights. Only foundation-model proxies and display previews are limited to
SDR sRGB. No sampled camera LUT is used.

**Input range** controls the video decoder's transport levels. Auto follows
container metadata; Full and Legal override that interpretation for incorrectly
labelled sources. It does not apply a second legal-range expansion after the
log curve. The transfer functions expect the manufacturer's normalized camera
codes. YCbCr matrix decoding is separate from camera gamut conversion.
The decoder relies on the stream's matrix metadata or FFmpeg defaults; selecting
a camera gamut does not repair missing or incorrect YCbCr matrix tags. In
particular, Apple Log 2 YCbCr recordings require their specified BT.2020
non-constant-luminance matrix to be tagged correctly.

## Output choices

- **Processed working colour** keeps the existing output convention: display
  previews in sRGB, EXR in linear Rec.709 primaries, and the existing cutout
  display/linear choice.
- **Processed camera colour** transforms predicted foreground back through the
  inverse gamut matrix and the selected camera's encoding function. Despill
  modifies the foreground before this conversion. This is not restoration of
  RGB altered by inference or despill.
- **Original RGB + new alpha** takes native-resolution decoded source RGB and
  attaches the predicted matte. RGB bypasses model reconstruction and despill.
  Transparent pixels can retain background colour; this mode is useful when
  the original plate must retain its colour for compositing downstream.
  This mode requires EXR and automatically uses 32-bit float pixels, including
  for Legacy inputs, so adjacent decoded 16-bit source codes remain distinct.

Alpha remains straight and linear. It is never logarithmically encoded.
Checker previews are display previews, not camera-log deliverables.

Float EXR is the fidelity option for source-colour outputs. Half EXR and video
encoding reduce precision; ProRes is lossy. Source-colour EXR is explicitly
encoded in the selected camera profile, unlike the default linear EXR output.
The colour sidecar next to a render records each output's interpretation; keep
it with the files so the viewer can display old results using their saved
profile, even after the clip's current selection changes. Set the same input
colour space explicitly when importing source-colour outputs into another app.

## Profiles and references

Profiles pair a specific transfer function with a specific gamut. A brand name
alone is not sufficient. Curves are evaluated with the analytic functions in
[Colour 0.4.6](https://github.com/colour-science/colour/tree/v0.4.6), with explicit
parameters, or directly from the manufacturer's published equations.

| Family | Defined profiles | Reference |
| --- | --- | --- |
| ARRI | LogC3 EI800 / ARRI Wide Gamut 3; LogC4 / ARRI Wide Gamut 4 | [LogC3 VFX specification](https://www.arri.com/resource/blob/31918/66f56e6abb6e5b6553929edf9aa7483e/2017-03-alexa-logc-curve-in-vfx-data.pdf), [LogC4 specification](https://www.arri.com/resource/blob/278790/f3318e8c9c65617d8c5ca3f8b3e32051/2023-05-arri-logc4-specification-data.pdf) |
| RED | Log3G10 / REDWideGamutRGB | [Colour reference implementation](https://colour.readthedocs.io/en/v0.4.6/generated/colour.models.log_encoding_Log3G10.html) |
| Blackmagic Design | Film Gen 5 / Blackmagic Wide Gamut | [Colour reference implementation](https://colour.readthedocs.io/en/v0.4.6/generated/colour.models.oetf_BlackmagicFilmGeneration5.html) |
| Sony | S-Log3 / S-Gamut3; S-Log3 / S-Gamut3.Cine | [Sony equations](https://pro.sony/s3/cms-static-content/uploadfile/06/1237494271406.pdf), [Sony gamut data](https://pro.sony/s3/2019/01/24031021/4461641181.pdf) |
| Canon | Canon Log 2 / Cinema Gamut; Canon Log 3 / Cinema Gamut | [Colour reference implementation](https://colour.readthedocs.io/en/v0.4.6/generated/colour.models.log_encoding_CanonLog2.html) |
| Fujifilm | F-Log / F-Gamut; F-Log2 / F-Gamut; F-Log2 C / F-Gamut C | [Fujifilm F-Log2 C specification](https://dl.fujifilm-x.com/technical-data/F-Log2C_DataSheet_E_Ver.1.0.pdf) |
| Apple | Apple Log / BT.2020; Apple Log 2 / Apple Wide Gamut | [Apple Log profile](https://developer.apple.com/documentation/avfoundation/avcapturecolorspace/applelog), [Apple Log 2 white paper copy](https://github.com/thatcherfreeman/aces-transforms/blob/master/specs/Apple%20Log%202%20White%20Pper.pdf) |
| Samsung | Samsung Log / BT.2020 | [Samsung developer specifications](https://developer.samsung.com/mobile/samsung-log-video.html), [2025 white paper copy](https://github.com/thatcherfreeman/aces-transforms/blob/master/specs/Samsung%2BLog%2BWhite%2BPaper_En.pdf) |
| Panasonic | V-Log / V-Gamut | [Colour reference implementation](https://colour.readthedocs.io/en/v0.4.6/generated/colour.models.log_encoding_VLog.html) |
| DJI | D-Log / D-Gamut | [DJI white paper](https://assets.ctfassets.net/et769tc4wc1v/5FsWboWAvO3VNfIUzvWCFa/e852902687ccfbcd418979ee625a1324/D-Log_D-Gamut_Whitepaper.pdf) |

LogC3 EI800 is an explicitly chosen exposure index, not a substitute for every
LogC3 recording. DJI D-Log uses the published baseline curve; the vendor
specification describes additional exposure-index behaviour. **D-Log M is a
different profile and is not an alias for D-Log.** No approximate D-Log M curve
is provided. Apple Log 2 uses the same transfer as Apple Log with different
primaries. The published toe floors in Apple and Samsung curves remain in
place; reversibility only holds within the specified invertible domains.

Camera profiles describe already decoded RGB. RAW development remains the
responsibility of a suitable decoder/application. The known variable-ACT
DNxHR limitation also remains a decoder limitation; changing the selected
colour profile cannot repair those corrupt decoded frames. Export a supported
intermediate from Resolve first.

The first implementation applies camera selection in the standalone queue.
OFX receives pixels in Resolve's current pipeline colour space, which can
already include camera transforms. Configure the Resolve colour pipeline and
the plugin's input transfer consistently to avoid applying a transform twice.

## CLI example

For an ARRI LogC4 plate, save original decoded camera RGB with the inferred
matte in float EXR:

```sh
python infer.py --input plate.mov --output_dir keyed \
  --input_color_space arri_logc4_awg4 --color_output source_rgb \
  --outputs cutout --output_format exr --exr_dtype float
```

Use `--color_output processed_source` for reconstructed/despilled foreground
returned to the selected camera encoding. Default `processed_working` keeps
the established display-video/linear-EXR convention. `--input_range full` or
`--input_range legal` can override incorrect transport-level metadata.
