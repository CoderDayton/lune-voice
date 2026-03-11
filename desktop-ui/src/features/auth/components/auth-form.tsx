import { useStore } from "@tanstack/react-store";
import { useForm } from "@tanstack/react-form";
import { useHotkey } from "@tanstack/react-hotkeys";
import { Button } from "../../../components/ui/button";
import { authActions, authStore } from "../../../store/auth-store";
import { AuthInputField } from "./auth-input-field";

interface AuthFormValues {
  name: string;
  email: string;
  password: string;
}

const providerProfiles = {
  Google: { email: "google@lune.local", name: "Google user" },
  Apple: { email: "apple@lune.local", name: "Apple user" },
} as const;

export function AuthForm() {
  const mode = useStore(authStore, (state) => state.mode);
  const isSignup = mode === "signup";
  const copy = isSignup
    ? {
        title: "Create your account",
        description:
          "Set up a private desktop workspace for local voice typing, saved sessions, and vocabulary fixes.",
        action: "Create account",
        support: "Already have an account?",
        supportAction: "Log in",
      }
    : {
        title: "Welcome back",
        description:
          "Log in to your Lune Voice workspace and pick up where you left off.",
        action: "Log in",
        support: "New to Lune Voice?",
        supportAction: "Create an account",
      };

  const form = useForm({
    defaultValues: {
      name: "",
      email: "",
      password: "",
    } satisfies AuthFormValues,
    onSubmit: ({ value }) => {
      authActions.signIn(value.email.trim(), value.name.trim() || undefined);
    },
  });

  useHotkey(
    "Mod+Enter",
    (event) => {
      event.preventDefault();
      void form.handleSubmit();
    },
    { enabled: true, preventDefault: true },
  );

  function submitProvider(provider: keyof typeof providerProfiles) {
    const profile = providerProfiles[provider];
    authActions.signIn(profile.email, profile.name);
  }

  return (
    <div className="auth-form-shell">
      <div className="stack-copy">
        <h1 className="auth-form-title">{copy.title}</h1>
        <p className="body-copy-soft">{copy.description}</p>
      </div>

      <div className="auth-provider-stack">
        {Object.keys(providerProfiles).map((provider) => (
          <Button
            key={provider}
            type="button"
            variant="secondary"
            className="auth-provider-button"
            onClick={() => submitProvider(provider as keyof typeof providerProfiles)}
          >
            Continue with {provider}
          </Button>
        ))}
      </div>

      <div className="auth-divider">
        <span className="auth-divider-line" />
        <span className="auth-divider-text">or use email</span>
        <span className="auth-divider-line" />
      </div>

      <form
        className="auth-form-grid"
        onSubmit={(event) => {
          event.preventDefault();
          void form.handleSubmit();
        }}
      >
        {isSignup ? (
          <form.Field
            name="name"
            validators={{
              onChange: ({ value }) => (!value.trim() ? "Enter your name." : undefined),
            }}
          >
            {(field) => (
              <AuthInputField
                label="Name"
                placeholder="Your name"
                value={field.state.value}
                error={field.state.meta.isTouched ? (field.state.meta.errors[0] as string | undefined) ?? null : null}
                onChange={field.handleChange}
                onBlur={field.handleBlur}
              />
            )}
          </form.Field>
        ) : null}

        <form.Field
          name="email"
          validators={{
            onChange: ({ value }) => {
              if (!value.trim()) {
                return "Enter your email.";
              }
              if (!value.includes("@")) {
                return "Use a valid email address.";
              }
              return undefined;
            },
          }}
        >
          {(field) => (
            <AuthInputField
              label="Email"
              type="email"
              placeholder="you@example.com"
              value={field.state.value}
              error={field.state.meta.isTouched ? (field.state.meta.errors[0] as string | undefined) ?? null : null}
              onChange={field.handleChange}
              onBlur={field.handleBlur}
            />
          )}
        </form.Field>

        <form.Field
          name="password"
          validators={{
            onChange: ({ value }) => {
              if (!value.trim()) {
                return "Enter your password.";
              }
              if (value.trim().length < 8) {
                return "Use at least 8 characters.";
              }
              return undefined;
            },
          }}
        >
          {(field) => (
            <AuthInputField
              label="Password"
              type="password"
              placeholder="Enter your password"
              value={field.state.value}
              error={field.state.meta.isTouched ? (field.state.meta.errors[0] as string | undefined) ?? null : null}
              onChange={field.handleChange}
              onBlur={field.handleBlur}
            />
          )}
        </form.Field>

        <form.Subscribe selector={(state) => ({ canSubmit: state.canSubmit, isSubmitting: state.isSubmitting })}>
          {({ canSubmit, isSubmitting }) => (
            <Button type="submit" className="w-full" disabled={!canSubmit || isSubmitting}>
              {isSubmitting ? "Opening workspace..." : copy.action}
            </Button>
          )}
        </form.Subscribe>
      </form>

      <div className="auth-hotkey-row">
        <span className="auth-hotkey-pill">Mod+Enter submit</span>
        <span className="auth-hotkey-pill">Alt+1 log in</span>
        <span className="auth-hotkey-pill">Alt+2 sign up</span>
      </div>

      <p className="body-copy-soft">
        {copy.support}{" "}
        <Button
          type="button"
          variant="ghost"
          className="auth-inline-link"
          onClick={() => authActions.setMode(isSignup ? "login" : "signup")}
        >
          {copy.supportAction}
        </Button>
      </p>
    </div>
  );
}
