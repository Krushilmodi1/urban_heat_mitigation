# Open Buildings CSV - inspection report

- File: `395_buildings.csv` (1.2 MB) - read only, not modified
- Rows: 6,027
- Representation: **polygons**
- Coordinate reference system assumed: EPSG:4326 (Google Open Buildings publishes WGS 84 degrees)
- Latitude range: 22.6566 to 23.3931; longitude range: 72.1693 to 72.9863
- Covers the whole study area: True
- Confidence range: 0.600 to 0.980
- Area column range (m2): 39.9 to 742.0

## Column mapping

- latitude: 'latitude' (auto-detected)
- longitude: 'longitude' (auto-detected)
- area_m2: 'area_in_meters' (auto-detected)
- confidence: 'confidence' (auto-detected)
- geometry_wkt: 'geometry' (auto-detected)
- building_id: 'full_plus_code' (auto-detected)

## Columns, types and missing values

| Column | Type | Missing |
|---|---|---|
| latitude | float64 | 0 |
| longitude | float64 | 0 |
| area_in_meters | float64 | 0 |
| confidence | float64 | 0 |
| geometry | str | 1 |
| full_plus_code | str | 0 |

## Time coverage

- Date-like columns: none. Treated as a single static snapshot (no construction year).

## Geometry

- Polygon: 5,025
- MultiPolygon: 1
- Unparsable WKT: 0
- Invalid geometries repaired (make_valid): 1
- Computed UTM area vs CSV area column: median difference +0.00%, 90 % within 0.06% (n = 2,000 sampled)

## Filtering

- Below confidence threshold (0.65): 782
- Missing location / geometry: 219
- Outside city + 10 km buffer: 226
- Duplicates removed (by full_plus_code): 23

## Result

| zone        |   zone_area_km2 |   building_count |   buildings_per_km2 |   total_footprint_km2 |   footprint_density_pct |   mean_footprint_m2 |   median_footprint_m2 | representation   | area_source     |   confidence_min |
|:------------|----------------:|-----------------:|--------------------:|----------------------:|------------------------:|--------------------:|----------------------:|:-----------------|:----------------|-----------------:|
| city        |         462.009 |             3380 |               7.316 |                 0.936 |                   0.203 |             276.954 |               242.428 | polygons         | computed_utm43n |             0.65 |
| buffer_ring |        1108.07  |             1397 |               1.261 |                 0.384 |                   0.035 |             274.827 |               243.664 | polygons         | computed_utm43n |             0.65 |
| study_area  |        1570.08  |             4777 |               3.043 |                 1.32  |                   0.084 |             276.332 |               242.786 | polygons         | computed_utm43n |             0.65 |

## Limitations

- Buildings are assigned to a zone and grid cell by their centroid; a footprint crossing a cell edge is counted wholly in one cell.
- Points-only data give counts, not measured footprint areas.
- A single snapshot cannot show building change between years.
