import sys
from pathlib import Path
import numpy as np
import irsdk
import re
from irsdk import YAML_CODE_PAGE, YAML_TRANSLATER, CustomYamlSafeLoader, YamlReader
import yaml as _yaml

def _read_yaml(ibt):
    hdr = ibt._header
    data = bytes(ibt._shared_mem[hdr.session_info_offset : hdr.session_info_offset + hdr.session_info_len])
    yaml_src = re.sub(YamlReader.NON_PRINTABLE, "", data.translate(YAML_TRANSLATER).rstrip(b"\x00").decode(YAML_CODE_PAGE))
    yaml_src = re.sub(r"(\w+: )(,.*)", r'\1"\2"', yaml_src)
    return _yaml.load(yaml_src, Loader=CustomYamlSafeLoader) or {}

def load_channels(path_str, needed_substrings=None):
    p = Path(path_str)
    ibt = irsdk.IBT()
    ibt.open(str(p))
    try:
        info = _read_yaml(ibt)
        channels = {}
        for header in ibt._var_headers:
            name = header.name
            if header.count and header.count > 1: continue
            if needed_substrings and not any(s.lower() in name.lower() for s in needed_substrings): continue
            try:
                arr = np.asarray(ibt.get_all(name), dtype=np.float32)
                if arr.ndim == 1:
                    channels[name] = arr
            except:
                pass
        return channels, info
    finally:
        ibt.close()

# Representative files
reps = {
    "EARLY_NORMAL_08:09": r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 08-09-46.ibt",
    "PEAK_STIFF_11:34":   r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 11-34-31.ibt",
    "MID_RETREAT_15:31":  r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 15-31-50.ibt",
    "FINAL_17:23":        r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 17-23-57.ibt",
}

for label, f in reps.items():
    print(f"\n{'='*70}\n{label}: {Path(f).name}\n{'='*70}")
    try:
        chans, info = load_channels(f, needed_substrings=["ride","shock","wheel","lat","long","speed","throttl","brake","lap"])
        print(f"Relevant channels found: {len(chans)}")
        for name in sorted(chans.keys())[:40]:
            arr = chans[name]
            print(f"  {name}: len={len(arr)}, mean={arr.mean():.2f}, min={arr.min():.2f}, max={arr.max():.2f}, p95={np.percentile(arr,95):.2f}")
        # Lap count
        if "Lap" in chans:
            laps = int(chans["Lap"].max())
            print(f"\n  Laps in file: ~{laps}")
        # Rough best lap estimate if LapDistPct + LapTime or similar
        if "LapDistPct" in chans and "Lap" in chans:
            # Simple: find lap boundaries
            lap = chans["Lap"]
            dist = chans["LapDistPct"]
            lap_times = []
            for l in range(1, int(lap.max())+1):
                mask = (lap == l)
                if mask.sum() > 100:
                    # crude duration via samples (assume 60Hz)
                    lap_times.append(mask.sum() / 60.0)
            if lap_times:
                print(f"  Crude lap durations (s): min={min(lap_times):.1f}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
