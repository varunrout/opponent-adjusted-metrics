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

output "cloud_run_job_names" {
  description = "Names of the 3 Cloud Run Jobs (ingest -> transform -> analyse). No 4th model job exists."
  value = {
    ingest    = google_cloud_run_v2_job.oam_ingest.name
    transform = google_cloud_run_v2_job.oam_transform.name
    analyse   = google_cloud_run_v2_job.oam_analyse.name
  }
}

output "pipeline_workflow_name" {
  description = "Name of the Cloud Workflows orchestration chaining the 3 jobs (manual invocation only)."
  value       = google_workflows_workflow.oam_pipeline.name
}
