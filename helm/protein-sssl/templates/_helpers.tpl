{{/*
Expand the name of the chart.
*/}}
{{- define "protein-sssl.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "protein-sssl.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "protein-sssl.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "protein-sssl.labels" -}}
helm.sh/chart: {{ include "protein-sssl.chart" . }}
{{ include "protein-sssl.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- if .Values.global }}
global.deployment/region: {{ .Values.global.region }}
global.deployment/zone: {{ .Values.global.zone }}
global.deployment/environment: {{ .Values.global.environment }}
{{- range .Values.global.compliance.frameworks }}
compliance.framework/{{ . }}: "true"
{{- end }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "protein-sssl.selectorLabels" -}}
app.kubernetes.io/name: {{ include "protein-sssl.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "protein-sssl.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "protein-sssl.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Global deployment labels
*/}}
{{- define "protein-sssl.globalLabels" -}}
{{- if .Values.global }}
global.deployment/region: {{ .Values.global.region }}
global.deployment/zone: {{ .Values.global.zone }}
global.deployment/environment: {{ .Values.global.environment }}
global.deployment/mode: {{ .Values.global.deploymentMode }}
{{- if .Values.global.multiRegion.enabled }}
global.multiregion/enabled: "true"
global.multiregion/primary: {{ .Values.global.multiRegion.primaryRegion }}
{{- end }}
{{- if .Values.global.compliance.frameworks }}
{{- range .Values.global.compliance.frameworks }}
compliance.framework/{{ . }}: "true"
{{- end }}
{{- end }}
{{- if .Values.global.i18n.enabled }}
global.i18n/enabled: "true"
global.i18n/default-language: {{ .Values.global.i18n.defaultLanguage }}
{{- end }}
{{- if .Values.global.accessibility.enabled }}
global.accessibility/enabled: "true"
global.accessibility/wcag-level: {{ .Values.global.accessibility.wcagLevel }}
{{- end }}
{{- if .Values.global.cultural.enabled }}
global.cultural/enabled: "true"
{{- range .Values.global.cultural.regions }}
global.cultural/{{ . }}: "true"
{{- end }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Security context based on global settings
*/}}
{{- define "protein-sssl.securityContext" -}}
{{- if .Values.global.security.podSecurityStandards }}
{{- if eq .Values.global.security.podSecurityStandards "restricted" }}
allowPrivilegeEscalation: false
capabilities:
  drop:
  - ALL
readOnlyRootFilesystem: true
runAsNonRoot: true
runAsUser: 65534
seccompProfile:
  type: RuntimeDefault
{{- else if eq .Values.global.security.podSecurityStandards "baseline" }}
allowPrivilegeEscalation: false
capabilities:
  drop:
  - ALL
runAsNonRoot: true
runAsUser: 1000
{{- else }}
{{- toYaml .Values.securityContext }}
{{- end }}
{{- else }}
{{- toYaml .Values.securityContext }}
{{- end }}
{{- end }}

{{/*
Pod security context based on global settings
*/}}
{{- define "protein-sssl.podSecurityContext" -}}
{{- if .Values.global.security.podSecurityStandards }}
{{- if eq .Values.global.security.podSecurityStandards "restricted" }}
runAsNonRoot: true
runAsUser: 65534
runAsGroup: 65534
fsGroup: 65534
seccompProfile:
  type: RuntimeDefault
{{- else if eq .Values.global.security.podSecurityStandards "baseline" }}
runAsNonRoot: true
runAsUser: 1000
runAsGroup: 1000
fsGroup: 1000
{{- else }}
{{- toYaml .Values.podSecurityContext }}
{{- end }}
{{- else }}
{{- toYaml .Values.podSecurityContext }}
{{- end }}
{{- end }}

{{/*
Generate encryption configuration
*/}}
{{- define "protein-sssl.encryptionConfig" -}}
{{- if .Values.global.security.encryption }}
atRest:
  enabled: {{ .Values.global.security.encryption.atRest }}
  {{- if .Values.global.security.encryption.atRest }}
  provider: "kubernetes"
  keyRotation: {{ .Values.global.security.encryption.keyRotation }}
  {{- end }}
inTransit:
  enabled: {{ .Values.global.security.encryption.inTransit }}
  {{- if .Values.global.security.encryption.inTransit }}
  protocol: "TLS1.3"
  certificateProvider: "cert-manager"
  {{- end }}
{{- end }}
{{- end }}

{{/*
Generate compliance annotations
*/}}
{{- define "protein-sssl.complianceAnnotations" -}}
{{- if .Values.global.compliance }}
{{- range .Values.global.compliance.frameworks }}
compliance.framework/{{ . }}: "enabled"
{{- end }}
compliance.data-classification: {{ .Values.global.compliance.dataClassification }}
compliance.audit-logging: {{ .Values.global.compliance.auditLogging | quote }}
{{- if .Values.global.compliance.dataRetention.enabled }}
compliance.data-retention: {{ .Values.global.compliance.dataRetention.defaultPeriodDays | quote }}
{{- end }}
{{- if .Values.global.compliance.exportControl.enabled }}
compliance.export-control: "enabled"
{{- end }}
{{- end }}
{{- end }}

{{/*
Generate regional routing configuration
*/}}
{{- define "protein-sssl.regionalRouting" -}}
{{- if .Values.global.multiRegion.enabled }}
current-region: {{ .Values.global.region }}
primary-region: {{ .Values.global.multiRegion.primaryRegion }}
cross-region-replication: {{ .Values.global.multiRegion.crossRegionReplication | quote }}
data-sovereignty: {{ .Values.global.multiRegion.dataSovereignty }}
{{- range .Values.global.multiRegion.regions }}
region.{{ .name }}/zone: {{ .zone }}
region.{{ .name }}/data-residency: {{ .dataResidency | quote }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Generate accessibility metadata
*/}}
{{- define "protein-sssl.accessibilityMetadata" -}}
{{- if .Values.global.accessibility.enabled }}
accessibility.wcag/level: {{ .Values.global.accessibility.wcagLevel }}
accessibility.features/screen-reader: {{ .Values.global.accessibility.screenReaderSupport | quote }}
accessibility.features/keyboard-navigation: {{ .Values.global.accessibility.keyboardNavigation | quote }}
accessibility.features/high-contrast: {{ .Values.global.accessibility.highContrast | quote }}
{{- end }}
{{- end }}

{{/*
Generate cultural adaptation metadata
*/}}
{{- define "protein-sssl.culturalMetadata" -}}
{{- if .Values.global.cultural.enabled }}
cultural.scientific-notation: {{ .Values.global.cultural.scientificNotation }}
cultural.adapt-colors: {{ .Values.global.cultural.adaptColors | quote }}
cultural.adapt-communication: {{ .Values.global.cultural.adaptCommunication | quote }}
{{- range .Values.global.cultural.regions }}
cultural.region/{{ . }}: "enabled"
{{- end }}
{{- end }}
{{- end }}

{{/*
Generate i18n metadata
*/}}
{{- define "protein-sssl.i18nMetadata" -}}
{{- if .Values.global.i18n.enabled }}
i18n.default-language: {{ .Values.global.i18n.defaultLanguage }}
i18n.default-locale: {{ .Values.global.i18n.defaultLocale }}
i18n.rtl-support: {{ .Values.global.i18n.rtlSupport | quote }}
i18n.supported-languages: {{ join "," .Values.global.i18n.supportedLanguages }}
{{- end }}
{{- end }}