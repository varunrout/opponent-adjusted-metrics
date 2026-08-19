locals {
  common_labels = {
    environment = "production"
    managed_by  = "terraform"
    system      = "opponent-adjusted-metrics"
  }
}

resource "google_storage_bucket" "data" {
  name          = "${var.project_id}-data"
  project       = var.project_id
  location      = var.region
  storage_class = "STANDARD"

  deletion_policy             = "PREVENT"
  force_destroy               = false
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = false
  }

  soft_delete_policy {
    retention_duration_seconds = 604800
  }

  labels = local.common_labels

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_storage_bucket" "artifacts" {
  name          = "${var.project_id}-artifacts"
  project       = var.project_id
  location      = var.region
  storage_class = "STANDARD"

  deletion_policy             = "PREVENT"
  force_destroy               = false
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = false
  }

  soft_delete_policy {
    retention_duration_seconds = 604800
  }

  labels = local.common_labels

  lifecycle {
    prevent_destroy = true
  }
}

locals {
  bigquery_datasets = {
    oam_core = {
      friendly_name = "OAM Core"
      description   = "Canonical structured analytical data."
    }
    oam_features = {
      friendly_name = "OAM Features"
      description   = "Curated feature tables for model pipelines."
    }
    oam_ml = {
      friendly_name = "OAM ML"
      description   = "Model registry metadata and evaluation results."
    }
    oam_serving = {
      friendly_name = "OAM Serving"
      description   = "Curated production serving aggregates."
    }
  }
}

resource "google_bigquery_dataset" "datasets" {
  for_each = local.bigquery_datasets

  project       = var.project_id
  dataset_id    = each.key
  friendly_name = each.value.friendly_name
  description   = each.value.description
  location      = var.region

  deletion_policy            = "PREVENT"
  delete_contents_on_destroy = false
  labels                     = local.common_labels

  lifecycle {
    prevent_destroy = true
  }
}
