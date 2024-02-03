# Asteroid Classification and Clustering

## Problem Statement
This data science project, in response to the Jet Propulsion Laboratory of the California Institute of Technology, focuses on the binary classification of asteroids for the NEO (Near Earth Object) flag.

![Performance Metrics](https://i.imgur.com/MH9vquS.png)

## About the Subject
In Astronomy, the size and complexity of data are increasing, particularly in the study of asteroids. The project aims to classify asteroids and calculate their diameter to determine their characteristics. This information is crucial in identifying potentially hazardous asteroids for Earth. The research paper citing the asteroid dataset reports an accuracy near 99.99%.

## About the Data Set
- **File Name:** dataset.csv
- **File Type:** CSV
- **File Size:** 456.58 MB
- **Rows:** 958524
- **Columns:** 45

## Basic Column Definitions
- **SPK-ID:** Object primary SPK-ID
- **Object ID:** Object internal database ID
- **Object fullname:** Object full name/designation
- **pdes:** Object primary designation
- **name:** Object IAU name
- **NEO:** Near-Earth Object (NEO) flag
- **PHA:** Potentially Hazardous Asteroid (PHA) flag
- **H:** Absolute magnitude parameter
- **Diameter:** Object diameter (from equivalent sphere) km Unit
- **Albedo:** Geometric albedo
- **Diameter_sigma:** 1-sigma uncertainty in object diameter km Unit
- **Orbit_id:** Orbit solution ID
- **Epoch:** Epoch of osculation in modified Julian day form
- **Equinox:** Equinox of reference frame
- **e:** Eccentricity
- **a:** Semi-major axis au Unit
- **q:** Perihelion distance au Unit
- **i:** Inclination; angle with respect to x-y ecliptic plane
- **tp:** Time of perihelion passage TDB Unit
- **moid_ld:** Earth Minimum Orbit Intersection Distance au Unit

## Installation
Clone the project:
```bash
git clone https://github.com/AKA-SSH/Asteroid-Classification-and-Clustering.git
