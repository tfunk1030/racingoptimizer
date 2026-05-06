- Indexed controls: heave springs (0-8 front, 0-9 rear), torsion bars (0-18)
- Must convert physical N/mm <-> index before reading/writing
- Has separate heave dampers
- ARB uses letter indices (A-E + Disconnected)
- Pushrod param = PushrodLengthDelta (not Offset)

 front_size_labels=["Disconnected", "A", "B", "C", "D", "E"],
 rear_size_labels=["Disconnected", "A", "B", "C", "D", "E"],
 front_blade_count=5,
 rear_blade_count=5,

 damper
        ls_comp_range=(0, 40),
        ls_rbd_range=(0, 40),
        hs_comp_range=(0, 40),
        hs_rbd_range=(0, 40),
        hs_slope_range=(0, 11),
        hs_slope_rbd_range=(0, 11),

        garage_ranges=GarageRanges(
        damper_click=(0, 40),
        front_pushrod_mm=(-40.0, 40.0),
        rear_pushrod_mm=(-40.0, 40.0),
        front_heave_nmm=(0.0, 8.0),            # indexed 0-8 (not N/mm)
        rear_third_nmm=(0.0, 9.0),             # rear heave spring indexed 0-9 (no third spring)
        front_heave_perch_mm=(-150.0, 100.0),
        rear_third_perch_mm=(-150.0, 100.0),   # rear heave perch offset
        front_torsion_od_mm=(0.0, 18.0),        # indexed 0-18 (not mm)
        rear_spring_nmm=(0.0, 18.0),            # rear torsion bar OD indexed 0-18 (no coil spring)
        rear_spring_perch_mm=(0.0, 0.0),        # N/A — Ferrari has no rear coil spring perch
        arb_blade=(1, 5),
        # iRacing legality limits for Ferrari 499P
        camber_front_deg=(-2.9, 0.0),           # hard garage limit
        camber_rear_deg=(-1.9, 0.0),            # hard garage limit (iRacing GTP legal max)
        toe_front_mm=(-3.0, 3.0),
        toe_rear_mm=(-2.0, 3.0),
        torsion_bar_turns_range=(-0.250, 0.250),  # Ferrari has torsion bar turns at all 4 corners
        brake_bias_migration=(-6.0, 6.0),
        diff_clutch_plates_options=[2, 4, 6],
        front_diff_preload_nm=(-50.0, 50.0),    # Ferrari has front AND rear diffs
        front_diff_preload_step_nm=5.0,
        heave_spring_resolution_nmm=1.0,        # indexed: step by 1
        rear_spring_resolution_nmm=1.0,         # rear torsion bar OD: step by 1
        front_heave_perch_resolution_mm=0.5,
        rear_third_perch_resolution_mm=0.5,
        torsion_bar_turns_resolution=0.125,