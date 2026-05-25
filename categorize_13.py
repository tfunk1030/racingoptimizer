import sys
from pathlib import Path
import irsdk
import re
from irsdk import YAML_CODE_PAGE, YAML_TRANSLATER, CustomYamlSafeLoader, YamlReader
import yaml as _yaml
from datetime import datetime

def _read_yaml(ibt):
    hdr = ibt._header
    data = bytes(ibt._shared_mem[hdr.session_info_offset : hdr.session_info_offset + hdr.session_info_len])
    yaml_src = re.sub(YamlReader.NON_PRINTABLE, "", data.translate(YAML_TRANSLATER).rstrip(b"\x00").decode(YAML_CODE_PAGE))
    yaml_src = re.sub(r"(\w+: )(,.*)", r'\1"\2"', yaml_src)
    return _yaml.load(yaml_src, Loader=CustomYamlSafeLoader) or {}

def get_key_setup(path_str):
    p = Path(path_str)
    ibt = irsdk.IBT()
    ibt.open(str(p))
    try:
        info = _read_yaml(ibt)
        setup = info.get("CarSetup", {}) or {}
        chassis = setup.get("Chassis", {}) or {}
        tires = setup.get("TiresAero", {}) or {}
        systems = setup.get("Systems", {}) or {}
        damp = setup.get("Dampers", {}) or {}

        front = chassis.get("Front", {}) or {}
        rear = chassis.get("Rear", {}) or {}
        aero = tires.get("AeroSettings", {}) or {}
        fuel = systems.get("Fuel", {}) or {}

        # Static RH
        lf_rh = (chassis.get("LeftFront", {}) or {}).get("RideHeight")
        lr_rh = (chassis.get("LeftRear", {}) or {}).get("RideHeight")

        result = {
            "file": p.name,
            "heave_f": front.get("HeaveSpring"),
            "heave_r": rear.get("HeaveSpring"),
            "perch_f": front.get("HeavePerchOffset"),
            "perch_r": rear.get("HeavePerchOffset"),
            "wing": aero.get("RearWingAngle"),
            "fuel": fuel.get("FuelLevel"),
            "static_rh_f": lf_rh,
            "static_rh_r": lr_rh,
        }
        # Quick damper summary (front heave LS/HS comp as proxy for stiffness)
        fh = damp.get("FrontHeave", {}) or {}
        result["damper_fh_lscomp"] = fh.get("LsCompDamping")
        result["damper_fh_hscomp"] = fh.get("HsCompDamping")
        return result
    finally:
        ibt.close()

# All 13 Belle Isle Acura files
files = [
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 07-42-30.ibt",
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 08-09-46.ibt",
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 10-09-08.ibt",
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 10-25-41.ibt",
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 10-58-13.ibt",
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 11-19-44.ibt",
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 11-34-31.ibt",
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 11-58-30.ibt",
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 12-53-00.ibt",
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 13-25-15.ibt",
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 15-31-50.ibt",
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 16-34-39.ibt",
    r"ibtfiles\acuraarx06gtp_belleisle 2026-05-24 17-23-57.ibt",
]

print("TIME | HEAVE F/R (N/mm) | PERCH F/R (mm) | WING | STATIC RH F/R (mm) | NOTES")
print("-" * 110)
for f in files:
    try:
        r = get_key_setup(f)
        # Parse time from filename
        ts = r["file"].split(" ")[1].replace(".ibt","").replace("-",":")
        print(f"{ts} | {r['heave_f']} / {r['heave_r']} | {r['perch_f']} / {r['perch_r']} | {r['wing']} | {r['static_rh_f']} / {r['static_rh_r']} | LScompFH={r.get('damper_fh_lscomp')}")
    except Exception as e:
        print(f"ERROR on {f}: {e}")
