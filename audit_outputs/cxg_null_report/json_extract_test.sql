SELECT
  JSON_VALUE(feature_values_json) AS inner_json_prefix,
  JSON_VALUE(PARSE_JSON(JSON_VALUE(feature_values_json)), '$.possession_start_x') AS possession_start_x,
  JSON_VALUE(PARSE_JSON(JSON_VALUE(feature_values_json)), '$.possession_start_zone') AS possession_start_zone
FROM `oam-varun-260819.oam_features.cxg_shot_features`
WHERE feature_values_json IS NOT NULL
LIMIT 1
