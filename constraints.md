# Setup Legalities

Hard bounds the optimizer must clamp every recommended parameter to. **Partial slice** — does not yet cover ARBs, dampers, corner weights, brake bias, diff, etc.

Car keys match `aero-maps/` filenames: `acura`, `bmw`, `cadillac`, `ferrari`, `porsche`. Per-car overrides shadow the defaults.

## Defaults (apply to all cars unless overridden)

### Rear wing angle
| min | max |
| --- | --- |
| 12.0° | 17.0° |

### Tyre cold pressure
| min | max |
| --- | --- |
| 165.0 kPa | 220.0 kPa |

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

## Per-car overrides

### acura
- **Rear wing angle:** 6.0° – 10.0° (different aerodynamic package)
