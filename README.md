# Tools for describing and manipulating demographic models.

A demographic model describes one or more demes (or populations),
how they change over time, and their relationships to one another.
`demes` provides a self-contained and unambiguous in-memory description
of a demographic model.

Please see the [documentation](https://popsim-consortium.github.io/demes-docs/main/index.html)
for more details.

## Goals
- A simple declarative high-level format for specifying demographic models. This format is 
  intended to be human-readable and to make it easy to correctly specify models.
- A corresponding low-level format that is an entirely unambiguous and portable description 
  of a model, which can be used by many different simulation frameworks as input. Thus,
  we can see part of the role of this package as *compiling* the high-level description of 
  a model into the corresponding low-level description.
- Robust validation of models and reporting of errors for invalid models.
