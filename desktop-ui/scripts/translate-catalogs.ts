#!/usr/bin/env bun
// SPDX-License-Identifier: GPL-3.0-or-later
/**
 * Fills in empty msgstr entries in Lingui PO catalog files using
 * embedded translation maps. Entries that already have a translation
 * are left untouched. Entries not in the map keep their empty msgstr
 * (Lingui falls back to the English msgid at runtime).
 */

import { readFileSync, writeFileSync } from "fs";
import { join } from "path";

const LOCALES_DIR = join(import.meta.dir, "../src/locales");
const NON_ENGLISH_LOCALES = [
  "es",
  "fr",
  "de",
  "pt",
  "it",
  "ja",
  "ko",
  "zh",
  "ru",
  "ar",
  "hi",
];

// ---------------------------------------------------------------------------
// Translation maps
// ---------------------------------------------------------------------------

const translations: Record<string, Record<string, string>> = {
  es: {
    "Recording mode": "Modo de grabación",
    "Voice input": "Entrada de voz",
    "Smart cleanup": "Limpieza inteligente",
    Settings: "Ajustes",
    Shortcuts: "Atajos",
    Recording: "Grabación",
    "Start dictation": "Iniciar dictado",
    "Stop dictation": "Detener dictado",
    "Cancel dictation": "Cancelar dictado",
    "Private on-device": "Privado en dispositivo",
    "Voice mode": "Modo de voz",
    "Recording style": "Estilo de grabación",
    "Text output": "Salida de texto",
    "Text delivery": "Entrega de texto",
    "Send method": "Método de envío",
    "Writing style": "Estilo de escritura",
    "Cleanup model": "Modelo de limpieza",
    "Cleanup sensitivity": "Sensibilidad de limpieza",
    "Automatic cleanup": "Limpieza automática",
    "Use hardware acceleration": "Usar aceleración por hardware",
    "Word-level alignment": "Alineación a nivel de palabra",
    "Adaptive pause detection": "Detección adaptativa de pausas",
    "Pause sensitivity": "Sensibilidad de pausa",
    "Keyboard controls": "Controles de teclado",
    "Start or stop recording": "Iniciar o detener grabación",
    "Cancel current recording": "Cancelar grabación actual",
    "API endpoint": "Punto de acceso API",
    "API key": "Clave API",
    "Search transcripts": "Buscar transcripciones",
    "Voice typing, simplified": "Dictado por voz, simplificado",
    "Lune Voice": "Lune Voice",
    "Private desktop dictation": "Dictado de escritorio privado",
    "Open settings": "Abrir ajustes",
    "Open dictionary": "Abrir diccionario",
    "Sign out": "Cerrar sesión",
    "Signed in": "Sesión iniciada",
    "Recent activity": "Actividad reciente",
    "Speaking time": "Tiempo hablado",
    "Visible words": "Palabras visibles",
    "Visible sessions": "Sesiones visibles",
    Vocabulary: "Vocabulario",
    "Add rule": "Agregar regla",
    Phrase: "Frase",
    Replacement: "Reemplazo",
    Action: "Acción",
    "Delete this word fix?": "¿Eliminar esta corrección?",
    Language: "Idioma",
    "Dismiss error": "Descartar error",
    Processing: "Procesando",
    "Audio activity": "Actividad de audio",
  },

  fr: {
    "Recording mode": "Mode d'enregistrement",
    "Voice input": "Entrée vocale",
    "Smart cleanup": "Nettoyage intelligent",
    Settings: "Paramètres",
    Shortcuts: "Raccourcis",
    Recording: "Enregistrement",
    "Start dictation": "Démarrer la dictée",
    "Stop dictation": "Arrêter la dictée",
    "Cancel dictation": "Annuler la dictée",
    "Private on-device": "Privé sur l'appareil",
    "Voice mode": "Mode vocal",
    "Recording style": "Style d'enregistrement",
    "Text output": "Sortie de texte",
    "Send method": "Méthode d'envoi",
    "Writing style": "Style d'écriture",
    "Cleanup model": "Modèle de nettoyage",
    "Automatic cleanup": "Nettoyage automatique",
    "Use hardware acceleration": "Utiliser l'accélération matérielle",
    "Word-level alignment": "Alignement au niveau des mots",
    "Adaptive pause detection": "Détection adaptative des pauses",
    "Pause sensitivity": "Sensibilité de pause",
    "Keyboard controls": "Contrôles clavier",
    "Start or stop recording": "Démarrer ou arrêter l'enregistrement",
    "Cancel current recording": "Annuler l'enregistrement en cours",
    "API endpoint": "Point d'accès API",
    "API key": "Clé API",
    "Search transcripts": "Rechercher les transcriptions",
    "Voice typing, simplified": "Saisie vocale, simplifiée",
    "Open settings": "Ouvrir les paramètres",
    "Open dictionary": "Ouvrir le dictionnaire",
    "Sign out": "Se déconnecter",
    "Recent activity": "Activité récente",
    "Speaking time": "Temps de parole",
    Vocabulary: "Vocabulaire",
    "Add rule": "Ajouter une règle",
    Phrase: "Expression",
    Replacement: "Remplacement",
    Action: "Action",
    "Delete this word fix?": "Supprimer cette correction ?",
    Language: "Langue",
    "Dismiss error": "Ignorer l'erreur",
    Processing: "Traitement",
    "Audio activity": "Activité audio",
  },

  de: {
    "Recording mode": "Aufnahmemodus",
    "Voice input": "Spracheingabe",
    "Smart cleanup": "Intelligente Bereinigung",
    Settings: "Einstellungen",
    Shortcuts: "Tastenkürzel",
    Recording: "Aufnahme",
    "Start dictation": "Diktat starten",
    "Stop dictation": "Diktat stoppen",
    "Cancel dictation": "Diktat abbrechen",
    "Private on-device": "Privat auf dem Gerät",
    "Voice mode": "Sprachmodus",
    "Recording style": "Aufnahmestil",
    "Text output": "Textausgabe",
    "Send method": "Sendemethode",
    "Writing style": "Schreibstil",
    "Cleanup model": "Bereinigungsmodell",
    "Automatic cleanup": "Automatische Bereinigung",
    "Use hardware acceleration": "Hardwarebeschleunigung verwenden",
    "Adaptive pause detection": "Adaptive Pausenerkennung",
    "Pause sensitivity": "Pausenempfindlichkeit",
    "Keyboard controls": "Tastatursteuerung",
    "Start or stop recording": "Aufnahme starten oder stoppen",
    "Cancel current recording": "Aktuelle Aufnahme abbrechen",
    "API endpoint": "API-Endpunkt",
    "API key": "API-Schlüssel",
    "Search transcripts": "Transkripte durchsuchen",
    "Voice typing, simplified": "Spracheingabe, vereinfacht",
    "Open settings": "Einstellungen öffnen",
    "Open dictionary": "Wörterbuch öffnen",
    "Sign out": "Abmelden",
    "Recent activity": "Letzte Aktivität",
    "Speaking time": "Sprechzeit",
    Vocabulary: "Vokabular",
    "Add rule": "Regel hinzufügen",
    Phrase: "Ausdruck",
    Replacement: "Ersetzung",
    Action: "Aktion",
    "Delete this word fix?": "Diese Wortkorrektur löschen?",
    Language: "Sprache",
    "Dismiss error": "Fehler schließen",
    Processing: "Verarbeitung",
    "Audio activity": "Audioaktivität",
  },

  ja: {
    "Recording mode": "録音モード",
    "Voice input": "音声入力",
    "Smart cleanup": "スマート修正",
    Settings: "設定",
    Shortcuts: "ショートカット",
    Recording: "録音",
    "Start dictation": "ディクテーション開始",
    "Stop dictation": "ディクテーション停止",
    "Cancel dictation": "ディクテーション取消",
    "Private on-device": "デバイス上でプライベート",
    "Voice mode": "音声モード",
    "Text output": "テキスト出力",
    "Send method": "送信方法",
    "Writing style": "文体",
    "Automatic cleanup": "自動修正",
    "Use hardware acceleration": "ハードウェアアクセラレーションを使用",
    "Adaptive pause detection": "適応型ポーズ検出",
    "Keyboard controls": "キーボード操作",
    "Start or stop recording": "録音の開始または停止",
    "Cancel current recording": "現在の録音を取消",
    "Search transcripts": "文字起こしを検索",
    "Voice typing, simplified": "音声入力をシンプルに",
    "Open settings": "設定を開く",
    "Sign out": "サインアウト",
    "Recent activity": "最近のアクティビティ",
    "Speaking time": "発話時間",
    Vocabulary: "辞書",
    "Add rule": "ルールを追加",
    Phrase: "フレーズ",
    Replacement: "置換",
    Action: "操作",
    "Delete this word fix?": "この修正を削除しますか？",
    Language: "言語",
    Processing: "処理中",
    "Audio activity": "音声アクティビティ",
  },

  zh: {
    "Recording mode": "录音模式",
    "Voice input": "语音输入",
    "Smart cleanup": "智能修正",
    Settings: "设置",
    Shortcuts: "快捷键",
    Recording: "录音",
    "Start dictation": "开始听写",
    "Stop dictation": "停止听写",
    "Cancel dictation": "取消听写",
    "Private on-device": "设备端私密",
    "Voice mode": "语音模式",
    "Text output": "文本输出",
    "Send method": "发送方式",
    "Writing style": "写作风格",
    "Automatic cleanup": "自动修正",
    "Use hardware acceleration": "使用硬件加速",
    "Adaptive pause detection": "自适应停顿检测",
    "Keyboard controls": "键盘控制",
    "Start or stop recording": "开始或停止录音",
    "Cancel current recording": "取消当前录音",
    "Search transcripts": "搜索转录",
    "Voice typing, simplified": "语音输入，化繁为简",
    "Open settings": "打开设置",
    "Sign out": "退出登录",
    "Recent activity": "最近活动",
    "Speaking time": "说话时长",
    Vocabulary: "词汇表",
    "Add rule": "添加规则",
    Phrase: "短语",
    Replacement: "替换",
    Action: "操作",
    "Delete this word fix?": "删除此修正？",
    Language: "语言",
    Processing: "处理中",
    "Audio activity": "音频活动",
  },

  ko: {
    "Recording mode": "녹음 모드",
    "Voice input": "음성 입력",
    "Smart cleanup": "스마트 정리",
    Settings: "설정",
    "Start dictation": "받아쓰기 시작",
    "Stop dictation": "받아쓰기 중지",
    "Cancel dictation": "받아쓰기 취소",
    "Private on-device": "기기 내 비공개",
    "Search transcripts": "녹취록 검색",
    "Voice typing, simplified": "음성 입력, 간편하게",
    "Sign out": "로그아웃",
    "Recent activity": "최근 활동",
    Vocabulary: "어휘",
    "Add rule": "규칙 추가",
    Language: "언어",
    Processing: "처리 중",
  },

  pt: {
    "Recording mode": "Modo de gravação",
    "Voice input": "Entrada de voz",
    "Smart cleanup": "Limpeza inteligente",
    Settings: "Configurações",
    "Start dictation": "Iniciar ditado",
    "Stop dictation": "Parar ditado",
    "Cancel dictation": "Cancelar ditado",
    "Private on-device": "Privado no dispositivo",
    "Search transcripts": "Pesquisar transcrições",
    "Voice typing, simplified": "Digitação por voz, simplificada",
    "Sign out": "Sair",
    "Recent activity": "Atividade recente",
    Vocabulary: "Vocabulário",
    "Add rule": "Adicionar regra",
    Language: "Idioma",
    Processing: "Processando",
  },

  it: {
    "Recording mode": "Modalità di registrazione",
    "Voice input": "Ingresso vocale",
    "Smart cleanup": "Pulizia intelligente",
    Settings: "Impostazioni",
    "Start dictation": "Avvia dettatura",
    "Stop dictation": "Ferma dettatura",
    "Cancel dictation": "Annulla dettatura",
    "Private on-device": "Privato sul dispositivo",
    "Search transcripts": "Cerca trascrizioni",
    "Sign out": "Esci",
    "Recent activity": "Attività recente",
    Vocabulary: "Vocabolario",
    "Add rule": "Aggiungi regola",
    Language: "Lingua",
    Processing: "Elaborazione",
  },

  ru: {
    "Recording mode": "Режим записи",
    "Voice input": "Голосовой ввод",
    "Smart cleanup": "Умная очистка",
    Settings: "Настройки",
    "Start dictation": "Начать диктовку",
    "Stop dictation": "Остановить диктовку",
    "Cancel dictation": "Отменить диктовку",
    "Private on-device": "Приватно на устройстве",
    "Search transcripts": "Поиск транскрипций",
    "Sign out": "Выйти",
    "Recent activity": "Недавняя активность",
    Vocabulary: "Словарь",
    "Add rule": "Добавить правило",
    Language: "Язык",
    Processing: "Обработка",
  },

  ar: {
    "Recording mode": "وضع التسجيل",
    "Voice input": "الإدخال الصوتي",
    "Smart cleanup": "التنظيف الذكي",
    Settings: "الإعدادات",
    "Start dictation": "بدء الإملاء",
    "Stop dictation": "إيقاف الإملاء",
    "Cancel dictation": "إلغاء الإملاء",
    "Private on-device": "خاص على الجهاز",
    "Search transcripts": "البحث في النصوص",
    "Sign out": "تسجيل الخروج",
    "Recent activity": "النشاط الأخير",
    Vocabulary: "المفردات",
    "Add rule": "إضافة قاعدة",
    Language: "اللغة",
    Processing: "جاري المعالجة",
  },

  hi: {
    "Recording mode": "रिकॉर्डिंग मोड",
    "Voice input": "वॉइस इनपुट",
    "Smart cleanup": "स्मार्ट सफ़ाई",
    Settings: "सेटिंग्स",
    "Start dictation": "डिक्टेशन शुरू करें",
    "Stop dictation": "डिक्टेशन रोकें",
    "Cancel dictation": "डिक्टेशन रद्द करें",
    "Private on-device": "डिवाइस पर निजी",
    "Search transcripts": "ट्रांसक्रिप्ट खोजें",
    "Sign out": "साइन आउट",
    "Recent activity": "हाल की गतिविधि",
    Vocabulary: "शब्दावली",
    "Add rule": "नियम जोड़ें",
    Language: "भाषा",
    Processing: "प्रोसेसिंग",
  },
};

// ---------------------------------------------------------------------------
// PO file parser / writer
// ---------------------------------------------------------------------------

/**
 * Parses a PO file and fills in empty msgstr values from the translation map.
 * Returns the updated file content as a string.
 *
 * Algorithm:
 *  - Walk lines one at a time tracking parser state.
 *  - Accumulate multi-line msgid continuation lines.
 *  - When we reach a `msgstr ""` line, check the map; if found, replace it.
 *  - All other lines pass through verbatim.
 */
function patchPo(content: string, map: Record<string, string>): { updated: string; filled: number } {
  const lines = content.split("\n");
  const out: string[] = [];

  let currentMsgid: string | null = null;
  let inMsgid = false;
  let inMsgstr = false;
  let filled = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.startsWith("msgid ")) {
      inMsgid = true;
      inMsgstr = false;
      // Extract the value between the outer quotes on this line.
      // e.g.  msgid "Hello"  →  Hello
      //       msgid ""       →  (empty — multi-line follows or header)
      currentMsgid = extractQuoted(line.slice("msgid ".length));
      out.push(line);
      continue;
    }

    if (line.startsWith("msgstr ")) {
      inMsgid = false;
      inMsgstr = true;

      const existing = extractQuoted(line.slice("msgstr ".length));

      // Only fill if currently empty AND we have a translation.
      if (existing === "" && currentMsgid !== null && currentMsgid !== "") {
        const translation = map[currentMsgid];
        if (translation !== undefined) {
          // Escape backslashes and double-quotes in the translation string.
          const escaped = translation.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
          out.push(`msgstr "${escaped}"`);
          filled++;
          continue;
        }
      }

      out.push(line);
      continue;
    }

    // Continuation lines start with `"` — they extend the preceding msgid/msgstr.
    if (line.startsWith('"') && (inMsgid || inMsgstr)) {
      if (inMsgid && currentMsgid !== null) {
        // Append the unescaped content to currentMsgid.
        currentMsgid += extractQuoted(line);
      }
      out.push(line);
      continue;
    }

    // Any other line (blank, comment, etc.) resets state.
    inMsgid = false;
    inMsgstr = false;
    out.push(line);
  }

  return { updated: out.join("\n"), filled };
}

/**
 * Extracts the string content between the outermost pair of double-quotes,
 * unescaping `\"` and `\\` sequences.
 *
 * Input examples:
 *   `"Hello world"`  →  `Hello world`
 *   `""`             →  ``
 *   `"Say \"hi\""` →  `Say "hi"`
 */
function extractQuoted(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed.startsWith('"') || !trimmed.endsWith('"')) return "";
  const inner = trimmed.slice(1, -1);
  // Unescape standard PO escape sequences.
  return inner.replace(/\\"/g, '"').replace(/\\\\/g, "\\").replace(/\\n/g, "\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

let totalFilled = 0;

for (const locale of NON_ENGLISH_LOCALES) {
  const poPath = join(LOCALES_DIR, locale, "messages.po");
  const map = translations[locale] ?? {};

  let content: string;
  try {
    content = readFileSync(poPath, "utf-8");
  } catch (err) {
    console.error(`  [${locale}] Could not read ${poPath}: ${err}`);
    continue;
  }

  const { updated, filled } = patchPo(content, map);

  if (filled > 0) {
    writeFileSync(poPath, updated, "utf-8");
    console.log(`  [${locale}] Filled ${filled} translation(s).`);
  } else {
    console.log(`  [${locale}] Nothing to fill (already translated or no matches).`);
  }

  totalFilled += filled;
}

console.log(`\nDone. Total entries filled: ${totalFilled}`);
