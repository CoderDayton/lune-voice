import { DictionaryForm } from "./components/dictionary-form";
import { DictionaryTable } from "./components/dictionary-table";

export function DictionaryPage() {
  return (
    <div className="page-stack">
      <DictionaryForm />
      <DictionaryTable />
    </div>
  );
}
