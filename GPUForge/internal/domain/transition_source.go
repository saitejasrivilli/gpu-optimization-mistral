package domain

// TransitionSource identifies what triggered a lifecycle transition. Required
// on every transition record per docs/lifecycle.md.
type TransitionSource string

const (
	SourceAgentReport      TransitionSource = "agent-report"
	SourceHealthMonitor    TransitionSource = "health-monitor"
	SourceOperator         TransitionSource = "operator"
	SourceScheduler        TransitionSource = "scheduler"
	SourceAdmissionControl TransitionSource = "admission-control"
	SourceClient           TransitionSource = "client"
)
