variable "project_id" {
  description = "Google Cloud project ID for OAM productionisation."
  type        = string
}

variable "region" {
  description = "Primary Google Cloud region."
  type        = string
  default     = "europe-west2"
}

variable "name_prefix" {
  description = "Prefix used for OAM cloud resource names."
  type        = string
  default     = "oam"
}

variable "pipeline_data_version" {
  description = "Pinned StatsBomb source commit SHA / data_version used across the ingest -> transform pipeline."
  type        = string
  default     = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
}
