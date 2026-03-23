import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return this.props.fallback ?? (
        <div className="flex items-center justify-center h-dvh p-4 sm:p-8">
          <div className="card bg-base-200 border border-error/20 max-w-sm sm:max-w-md w-full p-4 sm:p-6">
            <h2 className="text-base sm:text-lg font-semibold text-error">Something went wrong</h2>
            <p className="text-sm opacity-60 mt-2">{this.state.error.message}</p>
            <button
              type="button"
              className="btn btn-sm btn-ghost mt-4"
              onClick={() => this.setState({ error: null })}
            >
              Try again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
