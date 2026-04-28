# Racing Setup Optimization Engine — Vision & Prompt

## What This Is

A physics-based setup optimization engine for iRacing GTP Hypercar race cars. It ingests telemetry files (.ibt) and aerodynamic maps, builds empirical physics models from measured car behavior, and generates optimal setups by reasoning about trade-offs corner-by-corner and phase-by-phase.

## The Prompt

I have iRacing telemetry files (.ibt) and parsed aerodynamic maps for 5 GTP Hypercar race cars (BMW M Hybrid V8, Porsche 963, Cadillac V-Series.R, Acura ARX-06, Ferrari 499P). Each IBT file contains 60Hz time-series data for 100+ channels per lap: 4-corner shock deflections and velocities, ride heights, lateral and longitudinal acceleration, speed, brake pressure, throttle position, steering angle, yaw/pitch/roll rates, wheel speeds, tire temperatures and pressures, fuel level, downforce, and the complete garage setup that was used for that session.

I want you to build a physics-based setup optimization engine that works like this:

### 1. Data Ingestion — Use everything, lose nothing.

Parse every IBT file. Extract every channel at full 60Hz resolution. Process every completed lap individually — not session averages. Each lap is an independent observation of how this car behaved with this specific setup on this specific track in these specific conditions. Store the raw time-series data in a structured format that can be queried by car, track, setup configuration, corner, and lap.

### 2. Corner-Phase Decomposition — Think like an engineer.

For each lap, segment into individual corners using GPS/speed/lateral G. For each corner, decompose into phases: braking zone, trail-brake/entry, mid-corner (peak lateral load), exit/traction, and straight. For each phase, compute the actual physics state: understeer angle (from steering geometry vs lateral G), load transfer distribution (from shock deflection asymmetry), traction utilization (from wheel speed differentials), aero platform state (from ride height and pitch), roll angle and rate, damper velocities vs forces. This is the fundamental unit of analysis — not the lap, not the session.

### 3. Physics Model — Learn from data, not from textbooks.

Build an empirical physics model for each car from the measured data. Don't hardcode spring rate formulas or LLTD equations — fit them from what the car actually does. When you change the front heave spring from 30 to 50 N/mm, what ACTUALLY happens to front ride height, pitch angle, shock velocities, aero balance, and understeer at corner entry? Learn these relationships from the data across all sessions where this parameter varied. Every parameter interacts with every other parameter — model the coupled system, not independent channels. Use the aero maps to connect ride height changes to downforce changes. Use the measured shock behavior to connect spring/damper changes to platform dynamics.

### 4. Setup Evaluation — Quantify everything.

For a given setup, predict the car's behavior at every phase of every corner on the target track. Score each corner-phase on: grip utilization (how close to the limit), balance (neutral, understeer, or oversteer and by how much), stability (margin before loss of control), traction (power-down efficiency on exits), aero efficiency (drag vs downforce trade-off on straights), and platform control (ride height consistency, bottoming risk). The total evaluation is the weighted sum across all corners, where the weight is the time sensitivity of each corner (how much lap time a 1% improvement there is worth).

### 5. Optimization — Reason about trade-offs.

When recommending setup changes, think through the FULL consequence chain. Stiffening the front heave spring improves aero platform stability at high speed (benefiting fast corners and straights) but reduces mechanical compliance over bumps (hurting slow corners and braking zones). Quantify both sides using the per-corner-phase model. The optimal setup is the one that maximizes total lap performance across ALL corners, not the one that fixes one problem at the expense of others. Know that changing one parameter changes other things — stiffer springs change ride heights which change aero balance which change load transfer which change tire temps which change grip. Chase the chain.

### 6. Learning — Get smarter with every lap.

Every new IBT file is more training data. Re-fit the physics models. Track how prediction accuracy improves. Identify which parameter interactions are well-understood (many data points, consistent behavior) vs uncertain (sparse data, noisy). When the model is uncertain, be conservative. When it's confident, be aggressive. Report confidence alongside every recommendation. Over time, the system should converge on a complete understanding of each car's behavior and be able to generate optimal setups for any track from just the track model and aero maps.

### 7. Output — Justify every click.

For every parameter in the output setup, explain WHY that value was chosen: which corners benefit, which corners compromise, what's the net trade-off, what telemetry evidence supports it, and what would happen if you went ±1-2 clicks in either direction. The output should read like a race engineer's briefing, not a list of numbers.

## Key Data Assets

- **IBT Files:** 60Hz telemetry with 100+ channels per car, multiple sessions per track
- **Aero Maps:** Parsed downforce/drag surfaces as function of ride height and wing angle, per car
- **Track Models:** Corner segmentation, speed profiles, surface characteristics derived from IBT data
- **Cars:** BMW M Hybrid V8, Porsche 963, Cadillac V-Series.R, Acura ARX-06, Ferrari 499P — each with unique suspension architecture, spring types, damper layouts, and garage parameter sets

## What This Is NOT

- Not a lookup table of known-good setups
- Not a rule engine with hardcoded engineering formulas
- Not a system that summarizes laps into session averages and loses the detail
- Not a solver that treats parameters independently without understanding interactions
- Not a system that uses lap time as the primary optimization signal (lap time is the OUTCOME, not the INPUT — the input is measured physics at every corner)

## Philosophy

The aero maps give me downforce and drag as a function of ride height and wing angle for each car. The IBT files give me the measured reality of how each car actually behaves. Build the system that connects these two sources of truth into optimal setups.


### 8. User Experience — Simple commands, powerful output.

Running the optimizer should be dead simple from PowerShell. No complex flags, no remembering module paths, no wrapper scripts. Examples of what I want:

```powershell
# Analyze a session and get an optimal setup
optimize bmw sebring --wing 17

# Feed it new telemetry data
optimize learn "path\to\session.ibt"

# Get a setup for a new track using the learned model
optimize bmw laguna-seca --wing 16

# Compare two setups
optimize compare "session1.ibt" "session2.ibt"

# See what the model knows about a car
optimize status bmw
```

The CLI should auto-detect the car from the IBT filename, auto-detect the track, and default to sensible options. Power users can override with flags, but the default path should be: drop an IBT file in, get a setup out. Install should be `pip install .` and the command should just work. No `python -m pipeline.produce --car bmw --ibt "long/path/with spaces/file.ibt" --wing 17 --verbose --no-learn --free` — that's not user-friendly.


### 9. Track Model — Know every meter of every track.

Build a compounding track model for each track from ALL IBT data across ALL sessions. Every lap driven adds to the model. The track model should know:

**Surface characteristics at every point:**
- Where curbs are (detected from shock velocity spikes, pitch/roll transients, and ride height discontinuities at specific track positions)
- Where rumble strips are (periodic high-frequency shock inputs at consistent positions)
- Where bumps and surface irregularities are (shock velocity p99 mapped to track position)
- Surface grip variation (lateral G capability mapped to track position across many laps)
- Where off-track excursions happen (detected from sudden grip loss, wheel speed spikes, extreme shock events at non-standard positions)

**Corner characteristics:**
- Precise braking points, apex positions, and exit points for each corner (averaged and refined across hundreds of laps)
- Typical speed envelope at every point (min/median/max from all laps)
- Elevation changes and camber (derived from lateral/longitudinal G vs steering angle relationships)
- How each corner loads the car differently (which corners are front-limited, rear-limited, traction-limited, aero-limited)

**Data quality filtering:**
This is critical — when analyzing telemetry for physics model fitting, the track model tells you which data points to trust and which are noise. If the shock velocity spikes to 800 mm/s at a specific track position that the model knows is a curb strike, that data point should be tagged as "curb event" not used as evidence that the car needs stiffer springs. Off-track excursions, curb rides, rumble strip hits, and contact events should be identified and separated from clean racing data. The physics model should learn from clean laps and clean sections of messy laps — not throw away an entire lap because the driver took one curb aggressively.

**Compounding accuracy:**
The track model gets more accurate with every session. Early sessions establish the baseline. Subsequent sessions refine it — curb positions get more precise, surface grip maps get smoother, corner speed envelopes tighten. After 50+ laps the track model should know every meter of the circuit well enough to predict expected shock velocities, ride heights, and G-forces at any point, and flag anomalies that indicate driver error, setup problems, or data noise.


### 10. Weather & Track Conditions — Understand the environment.

iRacing IBT files contain real-time weather and track condition channels at 60Hz. The system should ingest and use ALL of them:

**Atmospheric conditions (time-series channels):**
- `AirTemp` — ambient air temperature (°C), affects air density → downforce and drag
- `AirDensity` — kg/m³, directly scales aerodynamic forces
- `AirPressure` — barometric pressure (mbar)
- `RelativeHumidity` — affects air density and tire behavior
- `WindVel` and `WindDir` — wind speed (m/s) and direction, affects straight-line aero balance and cornering asymmetry
- `FogLevel` — fog density

**Track surface (time-series channels):**
- `TrackTempCrew` — track surface temperature (°C), major driver of tire grip and degradation
- `TrackWetness` — track wetness level (0=dry to higher values for standing water)
- `WeatherDeclaredWet` — official wet flag
- `PrecipType` — precipitation type (none, drizzle, rain)
- `Skies` — sky condition (clear, partly cloudy, overcast)

**Session-level weather (YAML header):**
- WeatherType, atmospheric conditions at session start, track state

**How to use this data:**

The physics model should understand that the SAME setup behaves differently in different conditions:
- Higher track temp → more tire grip initially but faster degradation, lower tire pressures needed
- Higher air density → more downforce AND more drag at the same ride heights and wing angle — the aero maps need an air density correction factor
- Wind affects aero balance asymmetrically — headwind on a straight increases downforce going in but reduces it coming back, crosswind shifts aero balance left/right
- Track temp changes during a session (morning to afternoon) — the model should track how grip evolves and account for this when comparing laps from different points in a session
- Wet conditions fundamentally change everything — grip levels, braking distances, optimal lines, and setup priorities shift dramatically

When fitting the physics model, every data point should carry its environmental context. A shock deflection measurement at 25°C track temp is not the same as one at 40°C (tire stiffness changes with temperature, affecting load transfer paths). The model should learn these environmental sensitivities from the data rather than assuming conditions are constant.

When generating a setup for a target session, the user should be able to specify expected conditions (track temp, air temp, wind) and get a setup tuned for those conditions — not just the average of all conditions in the training data.
