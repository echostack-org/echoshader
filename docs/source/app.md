# Building an Echoshader application

Echoshader visualization elements can be composed into interactive Panel applications. The application layer is responsible for configuration, layout, controls, shared state, and interactions between elements.

> This page is an initial overview. The application API and examples will be expanded as the refactored app architecture is finalized.

## Application structure

An Echoshader application is built from:

- visualization elements such as echograms, tracks, histograms, and tables;
- a YAML configuration describing data sources, elements, controls, and layout;
- shared application state used to coordinate interactive components;
- a Panel application generated from the configuration.

This keeps the visualization elements independent from deployment and application-specific behavior.

## Building an application

A YAML configuration can be loaded and converted into a Panel application:

```python
from echoshader.app.builder import build_app
from echoshader.app.config import load_config

config = load_config("app_config.yaml")
app = build_app(config)
```

The resulting object is a Panel application and can be displayed in a notebook or served as a web application.

## Launching an application

Echoshader also provides a launcher:

```python
from echoshader.app import launch

launch("app_config.yaml")
```

The launcher builds the application from the YAML configuration and serves it using Panel.

## Interactive behavior

The application layer is where interactions between visualization elements are coordinated. Examples include:

- shared controls;
- linked echogram and track views;
- selections and box-selection behavior;
- coordinated updates between multiple elements;
- application layouts and tabs.

These interactions are intentionally kept separate from the standalone visualization elements.

## Deployment

Echoshader builds and launches the Panel application itself. Persistent execution, process supervision, and deployment infrastructure can be handled by the environment in which the application is deployed.

More complete application examples and configuration documentation will be added as the refactored application interface is finalized.
