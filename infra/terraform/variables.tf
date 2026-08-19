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
