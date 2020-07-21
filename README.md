# Tools for describing and manipulating demographic models.

A demographic model describes one or more demes (or populations),
how they change over time, and their relationships to one another.
`demes` provides a self-contained and unambiguous in-memory description
of a demographic model.

## Goals
- A simple declarative high-level format for specifying demographic models. This format is 
  intended to be human-readable and to make it easy to correctly specify models.
- A corresponding low-level format that is an entirely unambiguous and portable description 
  of a model, which can be used by many different simulation frameworks as input. Thus,
  we can see part of the role of this package as *compiling* the high-level description of 
  a model into the corresponding low-level description.
- Robust validation of models and reporting of errors for invalid models.
- Agnostic to choice of time-parameterisation (forwards, backwards,
  continuous, discrete).
- Incremental. A model can be constructed incrementally, remaining valid
  at each step of construction.
- Extensible. It should be easy to extend the in-memory description with
  additional properties.
- Mutable. When modifying an existing model, changes should cascade naturally
  through to dependent properties. Incompatibilities should be reported.
