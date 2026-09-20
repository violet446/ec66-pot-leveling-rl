# EC66 Newton sand simulation

This directory is an incremental Newton reimplementation of the Blender scene
in `C:\homework\real2sim2real\manipulator3_EC66\manipulator3_EC66`.

## Phase 3 (current)

Phase 3 adds a low-resolution implicit-MPM sand bed and one-way moving-collider
coupling:

- 30,304 particles clipped to the circular bin and initialized from the
  Blender height function;
- bulk density 1600 kg/m³, friction coefficient 0.68, Young's modulus 50 MPa
  and Poisson ratio 0.2 as explicit, provisional calibration parameters;
- MuJoCo advances the EC66 and supplies link poses/velocities to MPM;
- MPM moves against the scoop, robot, floor and wall collision proxies;
- the phase-3-only digging path smoothly lowers the tool by an additional
  0.12 m during Blender frames 29–118 so the proxy scoop visibly carries sand;
- sand-to-body impulses are collected as diagnostic wrenches and can be shown
  in the GL viewer, but are not applied back to the robot in this phase;
- fixed-grid CUDA graph capture for interactive performance on the RTX 4060.

Run the live simulation with:

```powershell
cd C:\newton\ec66_sand
.\run_phase3.bat
```

The viewer side panel includes **Show sand impulses**. The default nine-second
motion loops until the GL window is closed. A finite validation run is:

```powershell
$env:TMPDIR='C:\warp_tmp'
$env:TEMP='C:\warp_tmp'
$env:TMP='C:\warp_tmp'
$env:WARP_CACHE_PATH='C:\warp_cache'
python run_ec66_phase3.py --viewer null --num-frames 540 --no-loop --test --quiet
```

The visual digging depth can be adjusted without changing phases 1 or 2:

```powershell
.\run_phase3.bat --dig-depth-offset 0.15
```

Current full-run results:

- particles reach about 0.96 m during lifting, versus an initial maximum sand
  height of about 0.67 m;
- final particle z range after the complete cycle: 0.0991–0.7362 m;
- maximum particle radius from bin center: 1.1622 m (inside the 1.20 m wall);
- no NaN/Inf, no floor penetration and no wall escape;
- maximum measured one-way body reaction: 1812.8 N and 303.3 N·m;
- complete 540-frame headless run: about 20 s after kernel caching on this
  machine (faster than 60 Hz wall-clock simulation rate without rendering).

The material constants are physically plausible starting values, not calibrated
results. The reported reaction wrench is resolution- and parameter-dependent;
it must not yet be treated as a prediction of the real EC66 load.

Implementation patterns were adapted from Newton's installed
`example_mpm_granular.py`, `example_mpm_anymal.py`, and
`example_mpm_twoway_coupling.py` examples.

## Phase 2

Phase 2 adds:

- the 2.4 m inner-diameter bin from the Blender reference;
- a closed rigid bottom;
- 48 slightly overlapping wall collision proxies with consistent inward
  boundaries for later MPM use;
- MuJoCo contact generation and real-time contact visualization;
- a full-trajectory assertion that the robot and scoop do not hit the bin;
- a gravity/coriolis-compensated impedance controller whose torque command is
  clamped to the public URDF effort limits;
- an explicit 2.5 mm contact gap for robot and bin shapes. Newton's derived
  0.1 m default gap caused nonphysical early reactions near the rim.

Run it with:

```powershell
cd C:\newton\ec66_sand
.\run_phase2.bat
```

The segment count can be changed with, for example,
`run_phase2.bat --bin-segments 64`.

The interactive run opens Newton's GL window and loops the nine-second motion.
Contact normals can be displayed by the viewer. For repeatable checks without
opening a window, run:

```powershell
$env:TMPDIR='C:\warp_tmp'
$env:TEMP='C:\warp_tmp'
$env:TMP='C:\warp_tmp'
$env:WARP_CACHE_PATH='C:\warp_cache'
python validate_phase2_geometry.py
python run_ec66_phase2.py --viewer null --num-frames 540 --no-loop --test --quiet
```

Validated on the current machine with Newton 1.6.0 / Warp 1.17.0:

- all 431 half-frame reference poses: zero robot/bin contact;
- complete 9 s dynamic run at 60 Hz and 8 substeps: zero contact;
- maximum joint tracking errors: `[0.006, 2.074, 3.003, 7.863, 0.011,
  0.017]` degrees;
- peak commanded torques: `[0.435, 36.60, 19.16, 6.49, 0.013, 0.007]`
  N·m.

These are empty-bin checks. They validate geometry, timing and controller
stability, but they are not yet evidence of physically calibrated sand force.

## Phase 1

- simplified EC66 visual and collision primitives;
- public EC66 joint frames, limits, moving-link masses and inertias;
- the same half-frame IK trajectory used by the Blender source;
- MuJoCo position-drive tracking;
- real-time Newton GL display;
- no sand and no contact response yet.

Double-click `run_phase1.bat`, or run:

```powershell
cd C:\newton\ec66_sand
.\run_phase1.bat
```

The GL run loops continuously until the window is closed. Use the viewer's
pause control to inspect a pose.

For a finite headless validation run:

```powershell
python run_ec66_phase1.py --viewer null --num-frames 120 --no-loop
```

The current phase-1 gains are `kp=1200`, `kd=80`. They reduce reference
tracking lag while retaining the effort limits authored in the URDF; they are
still simulation settings, not identified EC66 servo gains.

## Parameter provenance

- Kinematics, joint limits and link inertials are based on the public Elite
  Robots EC66 URDF retained with the Blender project.
- Geometry is a phase-1 proxy made from cylinders, spheres and boxes.
- The scoop mass is provisionally 1 kg. It must be measured or obtained from
  CAD before torque/contact-force results are considered calibrated.
- Joint drive gains and damping are provisional numerical settings, not
  identified EC66 servo parameters.
- The public URDF masses total about 10.57 kg, versus the 17.5 kg whole-robot
  product figure. This discrepancy must be resolved during dynamics calibration.

## Planned phases

1. Validate Newton joint signs, TCP path and tracking error against Blender.
2. Add the bin and robust collision proxies. **Complete.**
3. Add low-resolution implicit-MPM sand with one-way robot-to-sand coupling. **Complete.**
4. Calibrate sand against bulk tests (density, repose angle and discharge).
5. Add MuJoCo/MPM two-way coupling and validate reaction forces/torques.
