# Setup Legalities

Hard bounds the optimizer must clamp every recommended parameter to. **Partial slice** — does not yet cover ARBs, dampers, corner weights, brake bias, diff, etc.

Car keys match `aero-maps/` filenames: `acura`, `bmw`, `cadillac`, `ferrari`, `porsche`. Per-car overrides shadow the defaults.

> **CRITICAL — only USER INPUT parameters belong here.** Several iRacing
> garage values look like inputs but are actually calculated readouts that
> the driver cannot type into the UI. The optimizer must never recommend a
> value for a calculated readout (that is the same correctness class of
> bug as a unit mismatch — the recommendation cannot be entered). Confirmed
> calculated readouts on GTP cars (do **not** add bounds for these):
>
> * **Static ride heights** (`Chassis.<corner>.RideHeight`) — output of
>   perch offsets, pushrod lengths, torsion-bar turns, spring rates,
>   corner weights. Adjust the inputs; ride height follows.
> * **Heave spring deflection** / **heave slider deflection**
>   (`Chassis.Front.HeaveSpringDefl` / `HeaveSliderDefl`, two-value strings
>   like `8.8 mm 97.7 mm`) — current/max deflection measured at the part.
>   The user-set spring rate (N/mm) and perch offset are the inputs.
> * **All `*Defl` and `*AtSpeed` fields**, the entire `AeroCalculator` block,
>   per-corner `CornerWeight`, `CrossWeight`, and `LastHotPressure` /
>   `LastTempsOMI` / `TreadRemaining`.
>
> The actual user inputs that drive ride height + heave platform (and need
> bounds added once captured from the iRacing garage UI):
> `HeavePerchOffset` (mm), `SpringPerchOffset` (mm),
> `PushrodLengthOffset` (mm), `TorsionBarTurns`, `TorsionBarOD`,
> `HeaveSpring` (N/mm rate), `SpringRate` (N/mm rate), `ThirdSpring` (N/mm).

## Defaults (apply to all cars unless overridden)

### Rear wing angle
| min | max |
| --- | --- |
| 12.0° | 17.0° |

### Tyre cold pressure
| min | max |
| --- | --- |
| 165.0 kPa | 220.0 kPa |

> **NOTE — calculated readouts.** The "Suspension deflections" and
> "Static ride height" sections below define observation envelopes for
> values the iRacing UI _calculates_ (`HeaveSpringDefl`, `HeaveSliderDefl`,
> `RideHeight`). The driver cannot type these values into the garage —
> they update as a consequence of perch offsets, pushrod lengths, and
> spring rates. The ontology marks them `user_settable=False` so the
> optimizer's search space and the briefing's "set this" output exclude
> them. The model still LEARNS the correlation between user inputs and
> these readouts (so it can score candidates). Keep the bounds for
> reference, but the recommender will never emit values for them.

### Suspension deflections
| parameter | min | max |
| --- | --- | --- |
| heave spring | 0.6 mm | 25.0 mm |
| heave slider | 25.0 mm | 45.0 mm |

### Static ride height
| corner | min | max |
| --- | --- | --- |
| front | 30.0 mm | 80.0 mm |
| rear  | 30.0 mm | 80.0 mm |

> **NOTE — estimated bounds.** The numeric defaults below are general
> estimates compiled from research on the iRacing GTP garage UI. They are
> NOT authoritative iRacing API values and DO drift across patches and
> per-car. The optimizer treats them as legal bounds (clamping every
> recommendation), so a too-narrow estimate suppresses exploration and a
> too-wide estimate produces values the UI rejects. Verify with the actual
> garage panel before trusting; per-car overrides under "Per-car overrides"
> below shadow these defaults.

### Heave spring rate
Front axle (these GTPs only have a front heave spring; the rear-axle
counterpart is the **rear third spring** below).
| min | max |
| --- | --- |
| 0.0 N/mm | 900.0 N/mm |

### Rear third spring rate
| min | max |
| --- | --- |
| 10.0 N/mm | 900.0 N/mm |

### Rear coil spring rate
| min | max |
| --- | --- |
| 100.0 N/mm | 300.0 N/mm |

### Heave perch offset front
Front-axle perch position. Drives the static front ride height.
| min | max |
| --- | --- |
| -100.0 mm | 100.0 mm |

### Spring perch offset rear
Per-corner rear perch. Drives the static rear ride height in concert with
the third perch and pushrod offsets.
| min | max |
| --- | --- |
| 25.0 mm | 45.0 mm |

### Third perch offset rear
Rear-axle third element perch position.
| min | max |
| --- | --- |
| 20.0 mm | 55.0 mm |

### Pushrod length offset front
| min | max |
| --- | --- |
| -40.0 mm | 40.0 mm |

### Pushrod length offset rear
| min | max |
| --- | --- |
| -40.0 mm | 40.0 mm |

### Anti-roll bar — front
ARB **blade index** (1 = softest, 5 = stiffest). ARB size selection (Soft /
Medium / Stiff) is a separate discrete control not yet modelled.
| min | max |
| --- | --- |
| 1 | 5 |

### Anti-roll bar — rear
| min | max |
| --- | --- |
| 1 | 5 |

### Damper — Low Speed Compression (LSC)
| corner | min | max |
| --- | --- | --- |
| FL | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| FR | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| RL | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| RR | <TODO: from iRacing UI> | <TODO: from iRacing UI> |

### Damper — High Speed Compression (HSC)
| corner | min | max |
| --- | --- | --- |
| FL | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| FR | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| RL | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| RR | <TODO: from iRacing UI> | <TODO: from iRacing UI> |

### Damper — Low Speed Rebound (LSR)
| corner | min | max |
| --- | --- | --- |
| FL | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| FR | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| RL | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| RR | <TODO: from iRacing UI> | <TODO: from iRacing UI> |

### Damper — High Speed Rebound (HSR)
| corner | min | max |
| --- | --- | --- |
| FL | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| FR | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| RL | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| RR | <TODO: from iRacing UI> | <TODO: from iRacing UI> |

### Corner weight (target)
| corner | min | max |
| --- | --- | --- |
| FL | <TODO: from iRacing UI> kg | <TODO: from iRacing UI> kg |
| FR | <TODO: from iRacing UI> kg | <TODO: from iRacing UI> kg |
| RL | <TODO: from iRacing UI> kg | <TODO: from iRacing UI> kg |
| RR | <TODO: from iRacing UI> kg | <TODO: from iRacing UI> kg |

### Brake bias
Percent of brake force going to the front axle (front:rear split).
The garage exposes a base bias (~47 % front for the GTPs), plus an
in-race **target offset** and **migration** in [-5, +5] click steps.
| min | max |
| --- | --- |
| 40.0 % | 60.0 % |

### Differential
Coast / power ratios are not yet modelled — they vary too much per car.
Preload is a single Nm scalar shared across the GTPs.
| parameter | unit | min | max |
| --- | --- | --- | --- |
| preload     | Nm | 0.0  | 150.0 |
| coast ratio | %  | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| power ratio | %  | <TODO: from iRacing UI> | <TODO: from iRacing UI> |

### Camber
Negative values only (top of wheel inboard). Front spans wider than rear.
| corner | min | max |
| --- | --- | --- |
| FL | -2.9 ° | 0.0 ° |
| FR | -2.9 ° | 0.0 ° |
| RL | -1.9 ° | 0.0 ° |
| RR | -1.9 ° | 0.0 ° |

### Toe
iRacing exposes toe in mm (per wheel) for the GTPs, not degrees, but the
constraints loader's "Toe" section is currently degree-based — kept
TODO until the units mismatch is resolved.
| corner | min | max |
| --- | --- | --- |
| FL | <TODO: units mismatch> ° | <TODO: units mismatch> ° |
| FR | <TODO: units mismatch> ° | <TODO: units mismatch> ° |
| RL | <TODO: units mismatch> ° | <TODO: units mismatch> ° |
| RR | <TODO: units mismatch> ° | <TODO: units mismatch> ° |

### Brake duct opening — front
| min | max |
| --- | --- |
| <TODO: from iRacing UI> | <TODO: from iRacing UI> |

### Brake duct opening — rear
| min | max |
| --- | --- |
| <TODO: from iRacing UI> | <TODO: from iRacing UI> |

### Throttle / brake mapping
Discrete iRacing setting; structure varies per car (curve preset, shape index, or pedal map). Loosely specced — fill once each car's UI is inspected.
| parameter | min | max |
| --- | --- | --- |
| <TODO: structure TBD per car> | <TODO: from iRacing UI> | <TODO: from iRacing UI> |

## Per-car overrides

> Per-car overrides shadow the defaults above. Lines must use the override
> grammar `- **<Section name>:** <min> – <max> <unit>` so the constraints
> loader can parse them.

### acura
- **Rear wing angle:** 6.0° – 10.0° (different aerodynamic package)
- **Heave spring rate:** 90.0 – 600.0 N/mm
- **Rear coil spring rate:** 60.0 – 300.0 N/mm

### bmw
- **Heave spring rate:** 30.0 – 100.0 N/mm
- **Rear third spring rate:** 100.0 – 300.0 N/mm
- **Rear coil spring rate:** 100.0 – 300.0 N/mm

### cadillac
- **Heave spring rate:** 20.0 – 200.0 N/mm
- **Rear third spring rate:** 100.0 – 1000.0 N/mm
- **Rear coil spring rate:** 105.0 – 300.0 N/mm

### porsche
- **Heave spring rate:** 150.0 – 600.0 N/mm
- **Rear third spring rate:** 80.0 – 800.0 N/mm
- **Rear coil spring rate:** 105.0 – 280.0 N/mm
- **Heave perch offset front:** 40.0 – 90.0 mm
