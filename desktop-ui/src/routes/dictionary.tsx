import { createFileRoute } from "@tanstack/react-router";
import { DictionaryPage } from "../features/dictionary/dictionary-page";

export const Route = createFileRoute("/dictionary")({
  component: DictionaryPage,
});
