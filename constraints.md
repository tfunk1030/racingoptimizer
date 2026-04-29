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

### Anti-roll bar — front
| min | max |
| --- | --- |
| <TODO: from iRacing UI> | <TODO: from iRacing UI> |

### Anti-roll bar — rear
| min | max |
| --- | --- |
| <TODO: from iRacing UI> | <TODO: from iRacing UI> |

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
| min | max |
| --- | --- |
| <TODO: from iRacing UI> % | <TODO: from iRacing UI> % |

### Differential
| parameter | unit | min | max |
| --- | --- | --- | --- |
| preload     | Nm | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| coast ratio | %  | <TODO: from iRacing UI> | <TODO: from iRacing UI> |
| power ratio | %  | <TODO: from iRacing UI> | <TODO: from iRacing UI> |

### Camber
| corner | min | max |
| --- | --- | --- |
| FL | <TODO: from iRacing UI> ° | <TODO: from iRacing UI> ° |
| FR | <TODO: from iRacing UI> ° | <TODO: from iRacing UI> ° |
| RL | <TODO: from iRacing UI> ° | <TODO: from iRacing UI> ° |
| RR | <TODO: from iRacing UI> ° | <TODO: from iRacing UI> ° |

### Toe
| corner | min | max |
| --- | --- | --- |
| FL | <TODO: from iRacing UI> ° | <TODO: from iRacing UI> ° |
| FR | <TODO: from iRacing UI> ° | <TODO: from iRacing UI> ° |
| RL | <TODO: from iRacing UI> ° | <TODO: from iRacing UI> ° |
| RR | <TODO: from iRacing UI> ° | <TODO: from iRacing UI> ° |

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

### acura
- **Rear wing angle:** 6.0° – 10.0° (different aerodynamic package)
