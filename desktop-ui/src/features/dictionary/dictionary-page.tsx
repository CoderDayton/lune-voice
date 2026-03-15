import { PageIntroPanel } from "../../components/ui/page-intro-panel";
import { DictionaryForm } from "./components/dictionary-form";
import { DictionaryRail } from "./components/dictionary-rail";
import { DictionaryTable } from "./components/dictionary-table";
import { useLingui } from "@lingui/react/macro";

export function DictionaryPage() {
  const { t } = useLingui();
  return (
    <div className="page-stack">
      <PageIntroPanel
        kicker="Vocabulary"
        title={t`Save the names and phrases you want Lune Voice to get right.`}
        description={t`Add common names, acronyms, and preferred spelling so repeated corrections happen automatically.`}
      />

      <DictionaryForm />

      <section className="layout-grid xl:grid-cols-[1.15fr_0.85fr]">
        <DictionaryTable />
        <DictionaryRail />
      </section>
    </div>
  );
}
