SELECT SUBSTR(TO_JSON_STRING(feature_values_json),1,800) AS sample_json FROM `oam-varun-260819.oam_features.cxg_shot_features` WHERE feature_values_json IS NOT NULL LIMIT 1
