resource "google_artifact_registry_repository" "oam_containers" {
  project       = var.project_id
  location      = var.region
  repository_id = "oam-containers"
  format        = "DOCKER"
  description   = "Artifact Registry repository that stores OAM production container images (api, dashboard, pipeline)."

  labels = local.common_labels

  docker_config {
    immutable_tags = true
  }

  deletion_policy = "PREVENT"

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.artifactregistry]
}
