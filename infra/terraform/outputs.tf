output "artifact_registry_repository_id" {
  description = "Artifact Registry repository ID"
  value       = google_artifact_registry_repository.oam_containers.repository_id
}

output "artifact_registry_location" {
  description = "Artifact Registry repository location"
  value       = google_artifact_registry_repository.oam_containers.location
}

output "pipeline_service_account_email" {
  description = "Email address of the pipeline runtime service account"
  value       = google_service_account.oam_pipeline_sa.email
}
