# Football Data Analysis

A collection of football data analysis and modelling projects. Each project lives
in its own folder with its own notebook, dependencies, and write-up, so they can
be read and run independently.

## Projects

| Project | Description |
|---|---|
| [expected-goals-model](expected-goals-model/) | An Expected Goals (xG) model built from scratch on StatsBomb open data. Starts from an interpretable logistic regression on shot distance and angle, then adds context and gradient boosting, with calibration and a benchmark against StatsBomb's own xG. |

More projects will be added over time, each in its own folder.

## Layout

Every project folder is self-contained. To run one, open its folder, install its
`requirements.txt`, and run its notebook. See the project's own README for the
exact steps.

## Data

Projects that use StatsBomb open data
(https://github.com/statsbomb/open-data) credit StatsBomb as required. See each
project for its specific data sources.
