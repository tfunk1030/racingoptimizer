# Setup Legalities

Hard bounds the optimizer must clamp every recommended parameter to. **Partial slice** — ARB blade indices, brake bias, differential preload, camber, springs, perches, pushrods, wing, tyre pressure, and observation envelopes for calculated platform readouts are covered. Damper clicks, corner-weight targets, toe, brake ducts, differential coast/power details, and throttle/brake mapping still have TODO bounds or units and are therefore blocked from recommendations until captured from the iRacing garage UI.

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
| 152.0 kPa | 220.0 kPa |

> **NOTE — calculated readouts.** The "Suspension deflections" and
> "Static ride height" sections below define observation envelopes for
> values the iRacing UI _calculates_ (`HeaveSpringDefl`, `HeaveSliderDefl`,
> `RideHeight`). The driver cannot type these values into the garage —
> they update as a consequence of perch offsets, pushrod lengths, and
> spring rates. The ontology marks them `user_settable=False` so the
> optimizer's search space and the briefing's "set this" output exclude
> them. The model learns the dynamic per-corner platform state from telemetry
> channels instead; these setup readout bounds are kept for reference, but the
> recommender will never emit values for them.

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

> **NOTE — damper click bounds.** 0..11 click range per corner per
> damper mode (LSC / HSC / LSR / HSR / HSC slope). iRacing's GTP garage
> exposes each as a discrete integer click count. Range captured from
> Cadillac's garage UI (`Cadillacbounds.md`) and applied as the default;
> other cars are assumed to follow the same envelope until verified.
> ARB-style integer rounding is enforced at the briefing render.

### Damper — Low Speed Compression (LSC)
| corner | min | max |
| --- | --- | --- |
| FL | 0 | 11 |
| FR | 0 | 11 |
| RL | 0 | 11 |
| RR | 0 | 11 |

### Damper — High Speed Compression (HSC)
| corner | min | max |
| --- | --- | --- |
| FL | 0 | 11 |
| FR | 0 | 11 |
| RL | 0 | 11 |
| RR | 0 | 11 |

### Damper — Low Speed Rebound (LSR)
| corner | min | max |
| --- | --- | --- |
| FL | 0 | 11 |
| FR | 0 | 11 |
| RL | 0 | 11 |
| RR | 0 | 11 |

### Damper — High Speed Rebound (HSR)
| corner | min | max |
| --- | --- | --- |
| FL | 0 | 11 |
| FR | 0 | 11 |
| RL | 0 | 11 |
| RR | 0 | 11 |

### Damper — High Speed Compression Slope (HS slope)
Separate click index from the four primary damper modes. Sets where the
damper transitions from low-speed to high-speed compression behaviour.
| corner | min | max |
| --- | --- | --- |
| FL | 0 | 11 |
| FR | 0 | 11 |
| RL | 0 | 11 |
| RR | 0 | 11 |

### Torsion bar turns
Per-side preload turn count on the front torsion bars. Cadillac and BMW
M Hybrid V8 expose this control on the front axle per BMWBounds.md /
Cadillacbounds.md; Ferrari has torsion bars at all 4 corners (RL/RR
rows below). Other cars use coil springs and ignore this section.
0.001 turn step on Cadillac/BMW; 0.125 turn step on Ferrari (per-car
override below).
| corner | min | max |
| --- | --- | --- |
| FL | -0.250 | 0.250 |
| FR | -0.250 | 0.250 |
| RL | -0.250 | 0.250 |
| RR | -0.250 | 0.250 |

### Torsion bar OD
Per-side outer-diameter selection on the front torsion bars (Cadillac
and BMW M Hybrid V8 expose the same 14-value list per BMWBounds.md /
Cadillacbounds.md). The garage UI's fixed list runs from 13.90 to 18.20
mm; the recommender clamps to the continuous envelope and the renderer
picks the nearest legal value via `ParameterSpec.discrete_values`.
Ferrari uses an integer index (0..18) instead of mm — bounded by the
per-car override below.
| corner | min | max |
| --- | --- | --- |
| FL | 13.90 | 18.20 |
| FR | 13.90 | 18.20 |
| RL | 13.90 | 18.20 |
| RR | 13.90 | 18.20 |

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

### Front differential preload
Ferrari 499P has a front diff in addition to the rear (most other GTPs
do not). Bound -50..+50 Nm per Ferraribounds.md, 5 Nm step. Cars
without a front diff don't list this parameter in their ontology, so
this default constraint is harmless dead-letter for them.
| min | max |
| --- | --- |
| -50.0 Nm | 50.0 Nm |

### Anti-roll bar size — front
Categorical iRacing UI selection. The ontology
(`ParameterSpec.choices`) carries the labels in stiffness-ascending
order: 0=Disconnect, 1=Soft, 2=Medium, 3=Stiff. DE searches the
integer-index envelope; the renderer maps back to the label.
| min | max |
| --- | --- |
| 0 | 3 |

### Anti-roll bar size — rear
Same encoding as the front ARB size (0..3 → Disconnect / Soft / Medium / Stiff).
| min | max |
| --- | --- |
| 0 | 3 |

### Differential coast/drive ramps
Coast/drive ramp angle pair selection. The ontology
(`ParameterSpec.choices`) carries the labels in lock-up-ascending
order: 0=40/65, 1=45/70, 2=50/75. DE searches the integer-index
envelope; the renderer maps back to the label string.
| min | max |
| --- | --- |
| 0 | 2 |

### Differential clutch friction plates
Discrete numeric (2 / 4 / 6 plates). The ontology's
`discrete_values=(2, 4, 6)` snaps the optimizer's continuous output to
the nearest legal value at render time.
| min | max |
| --- | --- |
| 2 | 6 |

### Camber
Negative values only (top of wheel inboard). Front spans wider than rear.
| corner | min | max |
| --- | --- | --- |
| FL | -2.9 ° | 0.0 ° |
| FR | -2.9 ° | 0.0 ° |
| RL | -1.9 ° | 0.0 ° |
| RR | -1.9 ° | 0.0 ° |

### Toe
Per-axle front (single scalar at `Chassis.Front.ToeIn`) and per-corner
rear (`Chassis.{Left,Right}Rear.ToeIn`) static toe in mm. The
optimizer trains the front + LR sides; iRacing UI requires LR=RR so
the renderer mirrors RR via `_MIRRORED_LEAVES`. Bounds per
Ferraribounds.md (BMWBounds.md just says "infinite", so same envelope
applies — the optimizer's confidence regime caps over-aggressive
extrapolation when the corpus has thin variance).
| corner | min | max |
| --- | --- | --- |
| front | -3.0 mm | 3.0 mm |
| RL | -2.0 mm | 3.0 mm |
| RR | -2.0 mm | 3.0 mm |

### Fuel level
Race fuel load (L). The iRacing GTP UI lets the user type any value
between roughly 1 L (effective minimum to reach pit lane) and the
tank cap. BMW M Hybrid V8 / Cadillac V-Series.R both run ~58 L for
race start; quali stints are user-set per track (typically 5..15 L
for 3 laps + reserve). The per-car constraint envelope here covers
both regimes; the `--quali` CLI flag is what biases the optimizer
toward an aggressive low-fuel setup and `--fuel N` pins the value.
| min | max |
| --- | --- |
| 1.0 L | 100.0 L |

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
- **Heave spring rate:** 0.0 – 900.0 N/mm
- **Rear third spring rate:** 0.0 – 900.0 N/mm
- **Rear coil spring rate:** 105.0 – 280.0 N/mm
- **Spring perch offset rear:** -100.0 – 100.0 mm
- **Third perch offset rear:** -100.0 – 100.0 mm

### cadillac
- **Heave spring rate:** 0.0 – 700.0 N/mm
- **Rear third spring rate:** 0.0 – 900.0 N/mm
- **Rear coil spring rate:** 105.0 – 280.0 N/mm
- **Spring perch offset rear:** -100.0 – 100.0 mm
- **Third perch offset rear:** -100.0 – 100.0 mm

### ferrari
Ferrari 499P diverges from BMW/Cadillac in several places per
Ferraribounds.md: indexed heave springs (front 0..8 / rear 0..9), wider
damper click range (0..40), wider perch envelopes (-150..+100 mm),
front diff in addition to rear, torsion bars at all 4 corners with a
0.125-turn step, and torsion bar OD as an integer index (0..18) instead
of the BMW/Cadillac 14-discrete-mm list.
- **Heave spring rate:** 0.0 – 8.0 index
- **Rear third spring rate:** 0.0 – 9.0 index
- **Heave perch offset front:** -150.0 – 100.0 mm
- **Third perch offset rear:** -150.0 – 100.0 mm
- **Damper Low Speed Compression:** 0 – 40 clicks
- **Damper High Speed Compression:** 0 – 40 clicks
- **Damper Low Speed Rebound:** 0 – 40 clicks
- **Damper High Speed Rebound:** 0 – 40 clicks
- **Torsion bar OD FL:** 0.0 – 18.0 index
- **Torsion bar OD FR:** 0.0 – 18.0 index
- **Torsion bar OD RL:** 0.0 – 18.0 index
- **Torsion bar OD RR:** 0.0 – 18.0 index
- **Anti-roll bar size front:** 0 – 5 index
- **Anti-roll bar size rear:** 0 – 5 index
- **Front differential preload:** -50.0 – 50.0 Nm

### porsche
- **Heave spring rate:** 150.0 – 600.0 N/mm
- **Rear third spring rate:** 80.0 – 800.0 N/mm
- **Rear coil spring rate:** 105.0 – 280.0 N/mm
- **Heave perch offset front:** 40.0 – 90.0 mm
