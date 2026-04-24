#!/usr/bin/env python3
"""Custom ONNX exporter for Qwen3-ASR-0.6B and the wav2vec2 forced aligner.

Bypasses optimum-cli entirely. Uses torch.onnx.export with manual wrappers
that correctly handle Qwen3-ASR's architecture (GQA, q_norm/k_norm, RoPE,
SwiGLU, tied embeddings) and produce clean FP16 graphs with proper KV cache
I/O.

Usage:
    python scripts/export_onnx.py --asr --output-dir models/qwen3-asr-0.6b-onnx
    python scripts/export_onnx.py --fa
    python scripts/export_onnx.py --asr --no-quantize

Requirements:
    pip install torch transformers onnx onnxscript onnxruntime
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import onnx
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", message="Missing annotation for parameter")
warnings.filterwarnings("ignore", category=FutureWarning, module="copyreg")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")

try:
    import qwen_asr  # noqa: F401
except ImportError:
    pass

DTYPE = torch.float16


# ── Utilities ────────────────────────────────────────────────────────────────


def _save_consolidated(model: onnx.ModelProto, out_path: Path) -> None:
    """Save ONNX model with all large tensors in a single external data file."""
    data_name = out_path.stem + ".onnx_data"
    data_path = out_path.parent / data_name
    if data_path.exists():
        data_path.unlink()

    onnx.save_model(
        model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
    )
    sz = data_path.stat().st_size / (1024 * 1024)
    print(f"    {data_name}: {sz:.1f} MB")


def _consolidate_external_data(directory: Path, onnx_name: str) -> None:
    onnx_path = directory / onnx_name
    if not onnx_path.exists():
        return
    print(f"  Consolidating {onnx_name}...")
    model = onnx.load(str(onnx_path), load_external_data=True)
    _save_consolidated(model, onnx_path)


def _list_files(directory: Path) -> None:
    total = 0
    for f in sorted(directory.iterdir()):
        sz = f.stat().st_size
        total += sz
        if sz > 1_000_000:
            print(f"  {f.name}: {sz / (1024 * 1024):.1f} MB")
    print(f"  TOTAL: {total / (1024 * 1024):.1f} MB")


# ── SmoothQuant Calibration ──────────────────────────────────────────────────

# Multilingual calibration corpus for SmoothQuant W8A8 quantization.  Covers
# English (~25%, dominant dictation use-case) plus balanced representation of
# all 30 Qwen3-ASR supported languages so the per-channel smoothing factors
# capture activation distributions across language-specific decoder pathways.
# ~160 samples.  Variety in register (formal, technical, casual, numbers) and
# sentence length exercises different sequence-length regimes.
_CALIB_TEXTS: list[str] = [
    # ── English (100) ───────────────────────────────────────────────────────
    "The meeting is scheduled for Tuesday at three pm.",
    "She finished the report ahead of schedule and her manager approved it.",
    "Install the dependency and then run the build command in the terminal.",
    "The quick brown fox jumps over the lazy dog.",
    "I wanted to follow up on our discussion last week regarding the project timeline.",
    "Let's make an async function called sync data that awaits a fetch to the API.",
    "Handle the case where the cache is empty or the key does not exist.",
    "So we gotta nuke the old user profile class and replace it with a new interface.",
    "Thank you for getting back to me. I will review the document and send my feedback by Friday.",
    "The main blocker is the dependency on the auth service owned by the platform team.",
    "From the respect paid her on all sides she seemed like a queen.",
    "The church was built in the fourteenth century and restored in eighteen ninety.",
    "The neural network was trained on a large corpus of annotated speech data.",
    "After three weeks of testing we finally shipped the feature to production.",
    "Hey can you take a look at the PR when you get a chance?",
    "The quarterly earnings report will be published on the company website.",
    "We need to refactor the authentication module before the next sprint.",
    "The patient was admitted to the hospital on December fifth with mild symptoms.",
    "Could you please mute yourself? There's a lot of background noise.",
    "Revenue increased by fourteen percent compared to the same quarter last year.",
    "The transformer architecture relies on multi-head self-attention mechanisms.",
    "I'll circle back with you after I get the numbers from the finance team.",
    "Add a try catch block around the database query to handle connection timeouts.",
    "Ladies and gentlemen, welcome aboard flight seven twenty three to London.",
    "The contract expires on March thirty first and we need to renegotiate the terms.",
    "She said she'd be about fifteen minutes late because of traffic on the highway.",
    "Please make sure the unit tests pass before you push your changes to main.",
    "Global temperatures have risen by approximately one point two degrees Celsius.",
    "The server returned a five hundred error when we tried to process the payment.",
    "I think the issue is in the middleware layer between the gateway and the backend.",
    "The new policy goes into effect on January first of next year.",
    "Can you read that back to me? I want to make sure I didn't miss anything.",
    "We observed a significant improvement in latency after switching to the new cache.",
    "Uh let me think about that for a second. Yeah I think we should go with option B.",
    "The study enrolled three hundred and forty two participants across eight sites.",
    "Set the environment variable to production and restart the Docker container.",
    "He spoke for roughly twenty minutes about the importance of renewable energy.",
    "Open a pull request against the develop branch and tag me as a reviewer.",
    "The population of the metropolitan area exceeded four point seven million.",
    "Don't forget to update the changelog and bump the version number.",
    # ── English slang / casual / internet speech ────────────────────────────
    "Bro that deploy was lowkey sketchy, we should probably roll it back.",
    "Nah I'm good, I already ate like an hour ago.",
    "That's cap, there's no way the tests passed on the first try.",
    "Yo did you see the new trailer? It looks fire honestly.",
    "I'm literally dying, the CI has been broken for three days straight.",
    "She really said that in the standup? That's wild bro.",
    "OK boomer, we don't use jQuery anymore.",
    "Fr fr that error message is so unhelpful it might as well not exist.",
    "It's giving spaghetti code vibes, we need to clean this up ASAP.",
    "No shot that runs in under two seconds, the query is way too heavy.",
    "Bet, I'll push the fix after lunch and ping you on Slack.",
    "Dude the latency on this endpoint is sus, something's definitely wrong.",
    "Deadass the whole service went down because of a missing semicolon.",
    "That's bussin, the new UI is way cleaner than the old one.",
    "Bruh moment when you realize you've been debugging the wrong branch.",
    "He's goated at system design, like actually cracked at it.",
    "I'm not gonna lie, the docs for this library are mid at best.",
    "Ayo who pushed to main without a review? That's not it chief.",
    "Slay, the refactor actually made the code readable for once.",
    "This meeting could've been an email, I'm just saying.",
    "Finna yeet this whole module and rewrite it from scratch.",
    "That PR is a vibe check for the entire codebase honestly.",
    "W take on the architecture decision, that's gonna save us months.",
    "L move putting the secrets in the repo, even on accident.",
    "Gonna dip early today, I've been grinding since like six AM.",
    "That feature request is so extra, nobody's ever gonna use it.",
    "Ight imma head out, my brain is cooked from this debugging session.",
    "Sheesh that function is like two hundred lines, we gotta break it up.",
    "Lowkey the intern wrote better code than half the senior devs.",
    "It's not a bug it's a feature, copium at its finest.",
    "Respectfully, that code review comment was kinda toxic.",
    "Touch grass bro, you've been staring at that stack trace for four hours.",
    "Main character energy walking into the sprint planning with zero tickets done.",
    "The API is ghosting us, keeps timing out with no error message.",
    "Tell me you don't write tests without telling me you don't write tests.",
    "Rent free in my head, I keep thinking about that race condition.",
    "That's a skill issue not a tooling issue, just read the docs.",
    "Unalive the process and restart it, the memory leak is getting bad.",
    "Big yikes on that SQL injection vulnerability, how did that pass review?",
    "NGL that's actually a pretty clean solution, props to whoever wrote it.",
    "Caught in four K pushing to prod on a Friday afternoon.",
    "POV you're the oncall engineer and everything breaks at three AM.",
    "Ratio'd by the linter, it found more issues than the reviewer did.",
    "This codebase is giving legacy vibes, nobody's touched it since twenty nineteen.",
    "Ate and left no crumbs with that database migration, zero downtime.",
    "The code is cheugy but it works so I'm not touching it.",
    "Periodt, we're not shipping without integration tests anymore.",
    "Iykyk but the real bottleneck is the network call not the algorithm.",
    "Mid opinion but I think we should just use Postgres for everything.",
    "Clapped back at the stakeholder with actual metrics, not just vibes.",
    "That commit message is giving unhinged energy, what does fix stuff even mean?",
    "Salty about the merge conflict, I literally just rebased five minutes ago.",
    "It's the lack of error handling for me, like what if the file doesn't exist?",
    "Based take, microservices are overkill for a team of three.",
    "Understood the assignment, shipped the hotfix in under twenty minutes.",
    "Core memory unlocked, remember when we used to FTP deploy to production?",
    "Living rent free in the tech debt backlog for six sprints now.",
    "Hits different when the tests actually catch a real bug for once.",
    "We stay winning, the migration finished with zero data loss.",
    "Hard pass on that framework, the bundle size is absolutely unhinged.",
    # ── Chinese (10) ────────────────────────────────────────────────────────
    "今天的会议讨论了季度业绩报告的初步结果和下一步计划。",
    "她打开窗户，清晨的阳光洒进了整个房间。",
    "请在周五之前提交你的项目进展报告，包括具体的数据指标。",
    "这个函数的时间复杂度是O(n log n)，可以满足性能要求。",
    "上海的常住人口已经超过了两千四百万人。",
    "我觉得这个方案可以，但是需要再考虑一下成本问题。",
    "新版本修复了三个关键的安全漏洞，建议尽快升级。",
    "下午三点半我们在会议室开一个简短的站会。",
    "深度学习模型的训练需要大量的标注数据和计算资源。",
    "麻烦帮我把这段话翻译成英文，谢谢。",
    # ── Cantonese (5) ───────────────────────────────────────────────────────
    "今日嘅會議傾咗好多關於項目進度嘅嘢。",
    "你幫我睇下呢個文件有冇問題先。",
    "聽日下晝三點喺辦公室開會，記得準時。",
    "呢個月嘅銷售額比上個月增長咗百分之十二。",
    "我哋要喺截止日期之前完成所有測試。",
    # ── Japanese (8) ────────────────────────────────────────────────────────
    "本日の会議では四���期業績レポートの初期結果について議論しました。",
    "彼女は窓を開けて、朝の光が部屋全体に差し込んだ。",
    "来週の月曜日までにレビューを完了してください。",
    "このアルゴリズムの計算量はO(n)で、十分高速です。",
    "東京の人口は約千四百万人で、世界最大級の都市圏です。",
    "すみません、もう一度言っていただけますか。",
    "新しいバージョンではユーザーインターフェースが大幅に改善されました。",
    "データベースのマイグレーションは深夜のメンテナンス時間に実行します。",
    # ── Korean (6) ──────────────────────────────────────────────────────────
    "오늘 회의에서 분기별 실적 보고서의 초기 결과를 논의했습니다.",
    "다음 주 화요일까지 코드 리뷰를 완료해 주세요.",
    "이 시스템의 응답 시간이 평균 이백 밀리초 이하로 줄었습니다.",
    "서울의 인구는 약 천만 명으로 한국 전체 인구의 오분의 일입니다.",
    "죄송합니다만 다시 한번 말씀해 주시겠어요?",
    "배포 전에 스테이징 환경에서 모든 테스트를 통과해야 합니다.",
    # ── French (8) ──────────────────────────────────────────────────────────
    "Je souhaite rappeler que ces droits étaient intégrés dans la réforme de deux mille huit.",
    "La prochaine réunion est prévue pour mardi à quinze heures.",
    "Le chiffre d'affaires a augmenté de dix-huit pour cent au troisième trimestre.",
    "Veuillez vérifier que tous les tests unitaires passent avant de fusionner.",
    "La température moyenne a augmenté d'environ un degré et demi depuis mille neuf cent.",
    "Est-ce que tu peux me renvoyer le fichier? Je ne le trouve plus.",
    "Le nouveau modèle utilise une architecture de type transformeur avec attention multi-tête.",
    "Paris compte environ deux millions d'habitants dans la ville intra-muros.",
    # ── German (8) ──────────────────────────────────────────────────────────
    "Es antwortet auf die Notwendigkeit, Filmförderung als Motor des interkulturellen Dialogs einzusetzen.",
    "Die Sitzung wurde auf nächsten Donnerstag verschoben.",
    "Der Umsatz ist im Vergleich zum Vorjahr um zwölf Prozent gestiegen.",
    "Bitte stellen Sie sicher, dass alle Abhängigkeiten korrekt installiert sind.",
    "Die Bevölkerung Berlins beträgt ungefähr drei Komma sieben Millionen Einwohner.",
    "Können Sie das bitte noch einmal wiederholen? Ich habe es nicht verstanden.",
    "Das neuronale Netz wurde mit über zwei Millionen annotierten Beispielen trainiert.",
    "Wir müssen den Fehler vor dem nächsten Release beheben.",
    # ── Spanish (8) ─────────────────────────────────────────────────────────
    "Quiero comenzar mis palabras haciendo referencia a la huelga general en defensa de la educación pública.",
    "El informe trimestral será publicado en el sitio web de la empresa.",
    "Los ingresos aumentaron un quince por ciento en comparación con el año anterior.",
    "Por favor, asegúrate de que las pruebas pasen antes de hacer el despliegue.",
    "La Ciudad de México tiene una población de aproximadamente nueve millones de habitantes.",
    "¿Puedes repetir eso? No te escuché bien por el ruido de fondo.",
    "El algoritmo de ordenamiento tiene una complejidad temporal de O(n log n).",
    "Necesitamos revisar el contrato antes de que expire el treinta y uno de marzo.",
    # ── Portuguese (5) ──────────────────────────────────────────────────────
    "A reunião foi adiada para a próxima semana por motivos de agenda.",
    "O faturamento cresceu dezenove por cento no último trimestre.",
    "Por favor, verifique se todos os testes estão passando antes de enviar.",
    "São Paulo é a cidade mais populosa do Brasil com mais de doze milhões de habitantes.",
    "O novo sistema reduziu o tempo de resposta em quarenta por cento.",
    # ── Italian (5) ─────────────────────────────────────────────────────────
    "Il G8 e il G20 hanno confermato il nostro indirizzo sull'abolizione del segreto bancario.",
    "La riunione è stata rinviata a giovedì prossimo.",
    "Il fatturato è aumentato del tredici per cento rispetto all'anno precedente.",
    "Assicurati che tutti i test siano superati prima di fare il rilascio.",
    "Roma ha una popolazione di circa due milioni e ottocentomila abitanti.",
    # ── Russian (5) ─────────────────────────────────────────────────────────
    "Заседание было перенесено на следующий четверг из-за расписания.",
    "Квартальный отчёт будет опубликован на сайте компании.",
    "Выручка выросла на шестнадцать процентов по сравнению с прошлым годом.",
    "Пожалуйста, убедитесь, что все зависимости установлены корректно.",
    "Население Москвы составляет около тринадцати миллионов человек.",
    # ── Arabic (5) ──────────────────────────────────────────────────────────
    "تم تأجيل الاجتماع إلى يوم الخميس القادم بسبب ظروف الجدول الزمني.",
    "ارتفعت الإيرادات بنسبة أربعة عشر بالمائة مقارنة بالعام الماضي.",
    "يرجى التأكد من اجتياز جميع الاختبارات قبل النشر.",
    "يبلغ عدد سكان القاهرة الكبرى حوالي عشرين مليون نسمة.",
    "النموذج الجديد يستخدم بنية المحولات مع آلية الانتباه الذاتي.",
    # ── Hindi (4) ───────────────────────────────────────────────────────────
    "आज की बैठक में तिमाही प्रदर्शन रिपोर्ट के प्रारंभिक परिणामों पर चर्चा हुई।",
    "कृपया सुनिश्चित करें कि सभी परीक्षण तैनाती से पहले सफल हों।",
    "दिल्ली की जनसंख्या लगभग दो करोड़ से अधिक है।",
    "नया संस्करण तीन प्रमुख सुरक्षा कमजोरियों को ठीक करता है।",
    # ── Thai (4) ────────────────────────────────────────────────────────────
    "การประชุมวันนี้ได้หารือเกี่ยวกับผลการดำเนินงานประจำไตรมาส",
    "กรุณาตรวจสอบว่าการทดสอบทั้งหมดผ่านก่อนการปรับใช้",
    "กรุงเทพมหานครมีประชากรประมาณสิบล้านคน",
    "รายได้เพิ่มขึ้นร้อยละสิบสองเมื่อเทียบกับปีที่แล้ว",
    # ── Vietnamese (4) ──────────────────────────────────────────────────────
    "Cuộc họp hôm nay đã thảo luận về kết quả hoạt động hàng quý.",
    "Vui lòng đảm bảo tất cả các bài kiểm tra đều đạt trước khi triển khai.",
    "Dân số thành phố Hồ Chí Minh vào khoảng chín triệu người.",
    "Doanh thu tăng mười bảy phần trăm so với cùng kỳ năm ngoái.",
    # ── Indonesian (4) ──────────────────────────────────────────────────────
    "Rapat hari ini membahas hasil kinerja kuartal dan rencana selanjutnya.",
    "Pastikan semua pengujian berhasil sebelum melakukan deploy ke produksi.",
    "Jakarta memiliki populasi sekitar sepuluh juta penduduk.",
    "Pendapatan meningkat sebesar lima belas persen dibandingkan tahun lalu.",
    # ── Turkish (4) ─────────────────────────────────────────────────────────
    "Toplantı gelecek perşembe gününe ertelendi.",
    "Gelir geçen yıla kıyasla yüzde on üç oranında arttı.",
    "Lütfen dağıtımdan önce tüm testlerin geçtiğinden emin olun.",
    "İstanbul'un nüfusu yaklaşık on altı milyon kişidir.",
    # ── Dutch (3) ───────────────────────────────────────────────────────────
    "De vergadering is verplaatst naar volgende donderdag.",
    "De omzet is met veertien procent gestegen ten opzichte van vorig jaar.",
    "Zorg ervoor dat alle tests slagen voordat je de wijzigingen pusht.",
    # ── Polish (3) ──────────────────────────────────────────────────────────
    "Spotkanie zostało przeniesione na przyszły czwartek z powodu harmonogramu.",
    "Przychody wzrosły o szesnaście procent w porównaniu z poprzednim rokiem.",
    "Proszę upewnić się, że wszystkie testy przechodzą przed wdrożeniem.",
    # ── Malay (3) ───────────────────────────────────────────────────────────
    "Mesyuarat hari ini membincangkan keputusan prestasi suku tahunan.",
    "Sila pastikan semua ujian lulus sebelum penggunaan ke produksi.",
    "Kuala Lumpur mempunyai populasi kira-kira dua juta penduduk.",
    # ── Swedish (3) ─────────────────────────────────────────────────────────
    "Mötet har flyttats till nästa torsdag på grund av schemat.",
    "Intäkterna ökade med elva procent jämfört med föregående år.",
    "Se till att alla tester går igenom innan du pushar ändringarna.",
    # ── Danish (2) ──────────────────────────────────────────────────────────
    "Mødet er udsat til næste torsdag.",
    "Omsætningen steg med tretten procent i forhold til sidste år.",
    # ── Finnish (2) ─────────────────────────────────────────────────────────
    "Kokous siirrettiin ensi torstaille aikataulusyistä.",
    "Liikevaihto kasvoi neljätoista prosenttia edellisvuoteen verrattuna.",
    # ── Czech (2) ───────────────────────────────────────────────────────────
    "Schůzka byla přesunuta na příští čtvrtek kvůli rozvrhu.",
    "Tržby vzrostly o dvanáct procent ve srovnání s předchozím rokem.",
    # ── Greek (2) ───────────────────────────────────────────────────────────
    "Η συνεδρίαση αναβλήθηκε για την επόμενη Πέμπτη.",
    "Τα έσοδα αυξήθηκαν κατά δεκατρία τοις εκατό σε σχέση με πέρυσι.",
    # ── Romanian (2) ────────────────────────────────────────────────────────
    "Ședința a fost amânată pentru joi viitoare din cauza programului.",
    "Veniturile au crescut cu paisprezece la sută față de anul trecut.",
    # ── Hungarian (2) ───────────────────────────────────────────────────────
    "Az ülést a jövő csütörtökre halasztották az időbeosztás miatt.",
    "A bevétel tizenkét százalékkal nőtt az előző évhez képest.",
    # ── Persian (2) ─────────────────────────────────────────────────────────
    "جلسه به پنجشنبه آینده موکول شد.",
    "درآمد نسبت به سال گذشته چهارده درصد افزایش یافت.",
    # ── Filipino (2) ────────────────────────────────────────────────────────
    "Ang pulong ay inilipat sa susunod na Huwebes dahil sa iskedyul.",
    "Ang kita ay tumaas ng labinlimang porsyento kumpara sa nakaraang taon.",
    # ── Macedonian (2) ──────────────────────────────────────────────────────
    "Состанокот беше одложен за следниот четврток поради распоредот.",
    "Приходите се зголемија за тринаесет проценти во споредба со минатата година.",
]






def _quantize_matmul_weights_int8(model: onnx.ModelProto) -> int:
    """Replace fp16 MatMul weights with INT8 via DequantizeLinear.

    For each MatMul with a constant weight W (fp16):
      scale = max(abs(W_fp32)) / 127  (per-tensor symmetric)
      W_i8 = round(clamp(W_fp32 / scale, -128, 127))
    Then replace: MatMul(x, W) → MatMul(x, DequantizeLinear(W_i8, scale))
    DequantizeLinear outputs fp16 so the rest of the graph is unchanged.
    """
    import numpy as np
    from onnx import numpy_helper
    import onnx.helper as oh

    inits = {i.name: i for i in model.graph.initializer}
    count = 0

    for node in list(model.graph.node):
        if node.op_type != "MatMul" or node.input[1] not in inits:
            continue
        wt_name = node.input[1]
        init = inits[wt_name]
        W = numpy_helper.to_array(init).astype(np.float32)

        scale = np.max(np.abs(W)) / 127.0
        if scale < 1e-10:
            continue
        W_i8 = np.clip(np.round(W / scale), -128, 127).astype(np.int8)
        scale_fp16 = np.float16(scale)

        i8_name = f"{wt_name}_i8"
        scale_name = f"{wt_name}_scale"
        dq_out = f"{wt_name}_dq"

        i8_init = numpy_helper.from_array(W_i8, name=i8_name)
        scale_init = numpy_helper.from_array(np.array(scale_fp16, dtype=np.float16), name=scale_name)
        model.graph.initializer.append(i8_init)
        model.graph.initializer.append(scale_init)

        dq_node = oh.make_node(
            "DequantizeLinear", inputs=[i8_name, scale_name], outputs=[dq_out],
        )

        # Insert DequantizeLinear before the MatMul.
        idx = list(model.graph.node).index(node)
        model.graph.node.insert(idx, dq_node)

        # Point MatMul at the dequantized output instead of raw weight.
        node.input[1] = dq_out

        # Remove old fp16 weight initializer.
        for i, ini in enumerate(model.graph.initializer):
            if ini.name == wt_name:
                model.graph.initializer.pop(i)
                break

        count += 1

    return count


def _smooth_model(
    model: onnx.ModelProto,
    act_maxes: dict[str, "np.ndarray"],
    alpha: float = 0.5,
) -> int:
    """Apply SmoothQuant transform in-place.  Returns number of smoothed pairs.

    Groups MatMuls that share the same norm output, computes a single combined
    scale per norm (element-wise max across all consumers' weight maxima), then
    applies once:  gamma /= s,  W *= s  for each consumer.
    """
    import numpy as np
    from collections import defaultdict
    from onnx import numpy_helper

    inits = {i.name: i for i in model.graph.initializer}
    out2node: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for o in node.output:
            out2node[o] = node

    # Group: gamma_name → [(act_name, wt_name), ...]
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    gamma_for_act: dict[str, str] = {}

    for node in model.graph.node:
        if node.op_type != "MatMul" or node.input[1] not in inits:
            continue
        act_name, wt_name = node.input[0], node.input[1]
        mul_node = out2node.get(act_name)
        if not mul_node or mul_node.op_type != "Mul":
            continue
        gamma_name = next((inp for inp in mul_node.input if inp in inits), None)
        if gamma_name is None or act_name not in act_maxes:
            continue
        groups[gamma_name].append((act_name, wt_name))
        gamma_for_act[act_name] = gamma_name

    count = 0
    for gamma_name, pairs in groups.items():
        a_max = act_maxes[pairs[0][0]].copy()
        a_max = np.clip(a_max, 1e-5, None)

        # Combined w_max: element-wise max across all consumer weights.
        combined_w_max = np.zeros_like(a_max)
        for _, wt_name in pairs:
            W = numpy_helper.to_array(inits[wt_name]).astype(np.float32)
            combined_w_max = np.maximum(combined_w_max, np.max(np.abs(W), axis=1))
        combined_w_max = np.clip(combined_w_max, 1e-5, None)

        scales = (a_max ** alpha) / (combined_w_max ** (1.0 - alpha))
        scales = np.clip(scales, 1e-5, None)

        # Apply to gamma once.
        gamma = numpy_helper.to_array(inits[gamma_name]).astype(np.float32)
        gamma_new = gamma / scales
        orig_gm_dtype = inits[gamma_name].data_type
        gm_np = gamma_new.astype(numpy_helper.to_array(inits[gamma_name]).dtype)
        inits[gamma_name].CopyFrom(numpy_helper.from_array(gm_np, name=gamma_name))
        inits[gamma_name].data_type = orig_gm_dtype

        # Apply per-weight scale to each consumer.
        for _, wt_name in pairs:
            W = numpy_helper.to_array(inits[wt_name]).astype(np.float32)
            W_new = W * scales[:, None]
            orig_wt_dtype = inits[wt_name].data_type
            wt_np = W_new.astype(numpy_helper.to_array(inits[wt_name]).dtype)
            inits[wt_name].CopyFrom(numpy_helper.from_array(wt_np, name=wt_name))
            inits[wt_name].data_type = orig_wt_dtype
            count += 1

    return count


def quantize_smoothquant(
    model_path: Path,
    embed_path: Path,
    model_id: str,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    skip_int8: bool = False,
) -> None:
    """SmoothQuant + optional INT8 weight quantization.  No external libs.

    1. Collect per-channel activation maxima at each Norm→MatMul boundary
    2. Compute per-channel scales s = act_max^α / weight_max^(1-α)
    3. Fold scales: divide norm gamma by s, multiply MatMul weight by s
    4. Replace fp16 MatMul weights with INT8 via DequantizeLinear
    """
    import numpy as np
    import onnxruntime as ort
    from onnx import numpy_helper
    from onnx import numpy_helper
    from transformers import AutoTokenizer

    alpha = 0.5
    print(f"  Quantizing {model_path.name} (SmoothQuant W8A8, α={alpha})...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    embed_sess = ort.InferenceSession(
        str(embed_path), providers=["CPUExecutionProvider"]
    )

    model = onnx.load(str(model_path))
    inits = {i.name for i in model.graph.initializer}
    pos_ndim = next(
        (len(i.type.tensor_type.shape.dim) for i in model.graph.input if i.name == "position_ids"), 2
    )

    # Find activation tensor names that feed from Mul (RMSNorm) into MatMul.
    out2node: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for o in node.output:
            out2node[o] = node

    act_tensors: set[str] = set()
    for node in model.graph.node:
        if node.op_type == "MatMul" and node.input[1] in inits:
            mul_node = out2node.get(node.input[0])
            if mul_node and mul_node.op_type == "Mul":
                act_tensors.add(node.input[0])

    print(f"    Found {len(act_tensors)} smoothable activation points")

    # Add activation tensors as extra graph outputs so ORT returns them.
    # Use FLOAT type — ORT casts fp16 intermediates to float for outputs.
    extra_outputs = []
    for name in act_tensors:
        vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT16, None)
        model.graph.output.append(vi)
        extra_outputs.append(name)

    # Serialize to bytes for ORT session (model already has data loaded).
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    output_names = [o.name for o in sess.get_outputs()]

    # Calibrate: collect per-channel max abs for each activation tensor.
    act_maxes: dict[str, np.ndarray] = {}
    n_samples = min(64, len(_CALIB_TEXTS))
    print(f"    Calibrating on {n_samples} samples...")

    for idx in range(n_samples):
        ids = tokenizer(_CALIB_TEXTS[idx], return_tensors="np").input_ids
        embeds = embed_sess.run(["embeds"], {"input_ids": ids})[0]
        seq = embeds.shape[1]
        pos = np.arange(seq, dtype=np.int64)
        pos_ids = (
            np.broadcast_to(pos, (3, 1, seq)).copy()
            if pos_ndim == 3
            else pos.reshape(1, seq)
        )
        feed = {
            "inputs_embeds": embeds,
            "position_ids": pos_ids,
            "attention_mask": np.ones((1, seq), dtype=np.int64),
        }
        for i in range(num_layers):
            feed[f"past_key_values.{i}.key"] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float16)
            feed[f"past_key_values.{i}.value"] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float16)

        results = sess.run(output_names, feed)
        result_map = dict(zip(output_names, results))

        for name in extra_outputs:
            if name not in result_map:
                continue
            val = np.abs(result_map[name].astype(np.float32))
            ch_max = val.reshape(-1, val.shape[-1]).max(axis=0)
            if name in act_maxes:
                act_maxes[name] = np.maximum(act_maxes[name], ch_max)
            else:
                act_maxes[name] = ch_max

    del sess

    # Remove extra outputs we added.
    for name in extra_outputs:
        for o in list(model.graph.output):
            if o.name == name:
                model.graph.output.remove(o)

    count = _smooth_model(model, act_maxes, alpha)
    print(f"    Smoothed {count} Norm→MatMul pairs")

    if not skip_int8:
        target_opset = 19
        cur_opset = next((o.version for o in model.opset_import if o.domain == ""), 17)
        if cur_opset < target_opset:
            model = onnx.version_converter.convert_version(model, target_opset)
        wq = _quantize_matmul_weights_int8(model)
        print(f"    Quantized {wq} MatMul weights to INT8")

    _save_consolidated(model, model_path)
    label = "SmoothQuant + W8" if not skip_int8 else "SmoothQuant (fp16)"
    print(f"  {model_path.name}: {label} complete")



# ── ASR Wrappers ──────────────────────────────────────────────────────────────


class EncoderWrapper(nn.Module):
    """ONNX-friendly audio encoder wrapper for batch=1 unpadded mel input.

    Mirrors Qwen3-ASR's chunked audio-tower path closely enough to preserve the
    exported encoder geometry. The upstream module uses Python list splits that
    do not survive ONNX export with dynamic mel lengths, so this wrapper keeps
    the same math but reformulates chunking/masking with tensor ops.
    """

    def __init__(self, thinker):
        super().__init__()
        at = thinker.audio_tower
        self.conv2d1 = at.conv2d1
        self.conv2d2 = at.conv2d2
        self.conv2d3 = at.conv2d3
        self.conv_out = at.conv_out
        self.register_buffer(
            "pos_embed", at.positional_embedding.positional_embedding
        )
        self.layers = at.layers
        self.ln_post = at.ln_post
        self.proj1 = at.proj1
        self.act = at.act
        self.proj2 = at.proj2
        self.n_window = int(at.n_window)
        self.n_window_infer = int(at.n_window_infer)
        self.chunk_size = int(at.n_window * 2)
        self.chunk_aftercnn = self._feat_extract_output_lengths(
            torch.tensor(self.chunk_size, dtype=torch.long)
        ).item()
        self.window_aftercnn = self.chunk_aftercnn * (self.n_window_infer // self.chunk_size)
        self.max_encoder_frames = 6000
        self.max_aftercnn = self._feat_extract_output_lengths(
            torch.tensor(self.max_encoder_frames, dtype=torch.long)
        ).item()
        self.max_attention_windows = (self.max_aftercnn + self.window_aftercnn - 1) // self.window_aftercnn

    def _feat_extract_output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        lengths = (lengths - 1) // 2 + 1
        lengths = (lengths - 1) // 2 + 1
        lengths = (lengths - 1) // 2 + 1
        return lengths

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runtime only uses a single unpadded utterance per encoder invocation.
        # Keep export faithful for that path instead of pretending to support
        # arbitrary padded batches.
        if mel.shape[0] != 1:
            raise ValueError("EncoderWrapper exports batch=1 only")

        total_frames = mel.shape[-1]
        feature_len = torch.ones(1, dtype=torch.long, device=mel.device) * total_frames
        chunk_size = self.chunk_size
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        padded_frames = num_chunks * chunk_size
        pad_frames = padded_frames - total_frames

        padded = torch.nn.functional.pad(mel, (0, pad_frames))
        chunks = padded.reshape(1, mel.shape[1], num_chunks, chunk_size).permute(2, 0, 1, 3).squeeze(1)

        chunk_lengths = torch.full(
            (num_chunks,),
            chunk_size,
            dtype=torch.long,
            device=mel.device,
        )
        chunk_lengths[-1] = torch.remainder(feature_len[0] - 1, chunk_size) + 1

        x = chunks.unsqueeze(1)  # [num_chunks, 1, 128, chunk_size]
        x = torch.nn.functional.gelu(self.conv2d1(x))
        x = torch.nn.functional.gelu(self.conv2d2(x))
        x = torch.nn.functional.gelu(self.conv2d3(x))
        b, c, f, t = x.size()
        x = self.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        x = x + self.pos_embed[:t, :].unsqueeze(0).to(x.dtype)

        chunk_aftercnn_lengths = self._feat_extract_output_lengths(chunk_lengths)
        valid_mask = (
            torch.arange(t, device=mel.device).unsqueeze(0) < chunk_aftercnn_lengths.unsqueeze(1)
        )
        hidden = x[valid_mask]

        aftercnn_len = self._feat_extract_output_lengths(feature_len)[0]
        step_ends = (
            torch.arange(1, self.max_attention_windows + 1, device=mel.device, dtype=torch.long)
            * self.window_aftercnn
        )
        step_ends = torch.minimum(
            step_ends,
            aftercnn_len.expand(self.max_attention_windows),
        )
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, device=mel.device, dtype=torch.long),
                step_ends,
            ]
        ).to(torch.int32)

        for layer in self.layers:
            hidden = layer(hidden, cu_seqlens)[0]
        hidden = self.ln_post(hidden)
        hidden = self.proj1(hidden)
        hidden = self.act(hidden)
        hidden = self.proj2(hidden)
        return hidden.unsqueeze(0)  # [1, S, D]


class EmbedTokensWrapper(nn.Module):
    def __init__(self, thinker):
        super().__init__()
        self.embed_tokens = thinker.model.embed_tokens

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class DecoderWrapper(nn.Module):
    """Wraps the Qwen3 decoder with KV cache for ONNX export.

    Accepts an explicit attention_mask so the graph supports batched inference
    (B>1 with padding). For B=1 single-item inference, pass all-ones mask.
    """

    def __init__(self, thinker, num_layers: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.text_model = thinker.model
        self.lm_head = thinker.lm_head
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def _build_causal_mask_4d(
        self,
        attention_mask: torch.Tensor,  # [B, total_len]
        seq_len: int,
        total_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a 4D causal attention mask from a 2D padding mask.

        Constructs the mask directly with triu + padding to avoid the
        non-traceable ``create_causal_mask`` in transformers ≥5.x.
        """
        past_len = total_len - seq_len
        # Upper-triangular causal mask: -inf above diagonal (relative to past_len)
        causal = torch.triu(
            torch.full((seq_len, total_len), torch.finfo(dtype).min, dtype=dtype,
                        device=attention_mask.device),
            diagonal=past_len + 1,
        )
        causal_4d = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, total_len]
        # Apply 2D padding mask: positions with 0 → -inf
        padding = (1.0 - attention_mask.to(dtype)).unsqueeze(1).unsqueeze(2)
        padding = padding * torch.finfo(dtype).min
        return causal_4d + padding  # [B, 1, seq_len, total_len]

    @torch.no_grad()
    def forward(
        self,
        inputs_embeds: torch.Tensor,       # [B, L, H]
        position_ids: torch.Tensor,         # [3, B, L] (mRoPE)
        attention_mask: torch.Tensor,       # [B, past_len + L]
        past_key_values_flat: list[torch.Tensor],
    ):
        from transformers import DynamicCache

        past = DynamicCache()
        for i in range(self.num_layers):
            k = past_key_values_flat[i * 2]
            v = past_key_values_flat[i * 2 + 1]
            past.update(k, v, i)

        seq_len = inputs_embeds.shape[1]
        total_len = attention_mask.shape[1]
        # Build 4D causal mask ourselves to bypass transformers' non-traceable
        # create_causal_mask. Passing a 4D mask makes the model skip its own.
        causal_mask_4d = self._build_causal_mask_4d(
            attention_mask, seq_len, total_len, inputs_embeds.dtype,
        )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=causal_mask_4d,
            past_key_values=past,
            use_cache=True,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        new_cache = outputs.past_key_values
        flat_present: list[torch.Tensor] = []
        for i in range(self.num_layers):
            k, v = new_cache[i]
            flat_present.append(k)
            flat_present.append(v)

        return (logits, flat_present)


# ── ASR Export ────────────────────────────────────────────────────────────────


def export_asr(
    output_dir: Path,
    model_id: str = "Qwen/Qwen3-ASR-0.6B",
    quantize: bool = True,
) -> None:
    """Export Qwen3-ASR-0.6B to three ONNX files (FP16), then SmoothQuant W8A8 decoder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ASR] Exporting {model_id} → {output_dir}")

    from transformers import AutoProcessor
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    full_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=DTYPE,
        attn_implementation="eager",
    )
    full_model.eval()

    thinker = full_model.thinker
    text_config = thinker.config.text_config
    num_layers = text_config.num_hidden_layers
    num_kv_heads = getattr(text_config, "num_key_value_heads", text_config.num_attention_heads)
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
    hidden_size = text_config.hidden_size

    print(
        f"[ASR] Architecture: {num_layers}L, {hidden_size}H, "
        f"{num_kv_heads}KV, {head_dim}D, dtype={DTYPE}"
    )

    # ── 1. Encoder ───────────────────────────────────────────────────────
    print("[ASR]   Exporting qwen3-asr-encoder.onnx ...")
    enc = EncoderWrapper(thinker)
    enc.eval()
    # Use a multi-chunk dummy (500 frames = 3 chunks of 200) so torch.export
    # traces the chunking/reshape/windowing logic with symbolic shapes rather
    # than baking in num_chunks=1 from a single-chunk dummy.
    dummy_mel = torch.randn(1, 128, 500, dtype=DTYPE)
    mel_dim = torch.export.Dim("mel_frames", min=2, max=6000)

    torch.onnx.export(
        enc,
        (dummy_mel,),
        str(output_dir / "qwen3-asr-encoder.onnx"),
        input_names=["mel"],
        output_names=["features"],
        dynamic_shapes=({2: mel_dim},),
    )
    print("[ASR]   qwen3-asr-encoder.onnx exported")

    # ── 2. Embed tokens ──────────────────────────────────────────────────
    print("[ASR]   Exporting qwen3-asr-embed-tokens.onnx ...")
    emb = EmbedTokensWrapper(thinker)
    emb.eval()
    dummy_ids = torch.tensor([[151644, 8948, 198]], dtype=torch.long)
    seq_dim = torch.export.Dim("seq_len", min=1, max=4096)

    torch.onnx.export(
        emb,
        (dummy_ids,),
        str(output_dir / "qwen3-asr-embed-tokens.onnx"),
        input_names=["input_ids"],
        output_names=["embeds"],
        dynamic_shapes=({1: seq_dim},),
    )
    print("[ASR]   qwen3-asr-embed-tokens.onnx exported")

    # ── 3. Decoder (dynamic batch + attention_mask for B>1 support) ────
    print("[ASR]   Exporting qwen3-asr-decoder.onnx ...")
    dec = DecoderWrapper(thinker, num_layers, num_kv_heads, head_dim)
    dec.eval()

    batch_size = 1
    seq_len = 10
    past_len = 0
    dummy_embeds = torch.randn(batch_size, seq_len, hidden_size, dtype=DTYPE)
    dummy_position_ids = (
        torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
    )
    # attention_mask covers full sequence (past + current)
    dummy_attn_mask = torch.ones(batch_size, past_len + seq_len, dtype=torch.long)
    dummy_past = []
    for _ in range(num_layers):
        dummy_past.append(torch.zeros(batch_size, num_kv_heads, past_len, head_dim, dtype=DTYPE))
        dummy_past.append(torch.zeros(batch_size, num_kv_heads, past_len, head_dim, dtype=DTYPE))

    input_names = ["inputs_embeds", "position_ids", "attention_mask"]
    output_names = ["logits"]
    for i in range(num_layers):
        input_names.append(f"past_key_values.{i}.key")
        input_names.append(f"past_key_values.{i}.value")
        output_names.append(f"present_key_values.{i}.key")
        output_names.append(f"present_key_values.{i}.value")

    # NOTE: Qwen3 decoder + DynamicCache is incompatible with torch.export
    # (dynamo path). Use legacy opset export with dynamic_axes instead.
    dynamic_axes: dict[str, dict[int, str]] = {
        "inputs_embeds": {1: "seq_len"},
        "position_ids": {2: "seq_len"},
        "attention_mask": {1: "attn_len"},
        "logits": {1: "seq_len"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_key_values.{i}.key"] = {2: "past_seq_len"}
        dynamic_axes[f"past_key_values.{i}.value"] = {2: "past_seq_len"}
        dynamic_axes[f"present_key_values.{i}.key"] = {2: "total_seq_len"}
        dynamic_axes[f"present_key_values.{i}.value"] = {2: "total_seq_len"}

    torch.onnx.export(
        dec,
        (dummy_embeds, dummy_position_ids, dummy_attn_mask, dummy_past),
        str(output_dir / "qwen3-asr-decoder.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    print("[ASR]   qwen3-asr-decoder.onnx exported")

    # ── 4. Consolidate ───────────────────────────────────────────────────
    for name in [
        "qwen3-asr-encoder.onnx",
        "qwen3-asr-embed-tokens.onnx",
        "qwen3-asr-decoder.onnx",
    ]:
        _consolidate_external_data(output_dir, name)

    # ── 5. Quantize ────────────────────────────────────────────────────
    if quantize:
        quantize_smoothquant(
            output_dir / "qwen3-asr-decoder.onnx",
            output_dir / "qwen3-asr-embed-tokens.onnx",
            model_id,
            num_layers,
            num_kv_heads,
            head_dim,
        )
    else:
        # --fp16-decoder: smooth decoder (quality) but keep fp16 weights.
        # Quantize encoder + embed-tokens to INT8 instead.
        quantize_smoothquant(
            output_dir / "qwen3-asr-decoder.onnx",
            output_dir / "qwen3-asr-embed-tokens.onnx",
            model_id,
            num_layers,
            num_kv_heads,
            head_dim,
            skip_int8=True,
        )
        for name in ["qwen3-asr-encoder.onnx", "qwen3-asr-embed-tokens.onnx"]:
            path = output_dir / name
            print(f"  Quantizing {name} weights to INT8...")
            m = onnx.load(str(path))
            target_opset = 19
            cur = next((o.version for o in m.opset_import if o.domain == ""), 17)
            if cur < target_opset:
                m = onnx.version_converter.convert_version(m, target_opset)
            n = _quantize_matmul_weights_int8(m)
            _save_consolidated(m, path)
            print(f"    {name}: {n} weights quantized")

    # ── 6. Cleanup ───────────────────────────────────────────────────────
    keep = {
        "qwen3-asr-encoder.onnx", "qwen3-asr-encoder.onnx_data",
        "qwen3-asr-embed-tokens.onnx", "qwen3-asr-embed-tokens.onnx_data",
        "qwen3-asr-decoder.onnx", "qwen3-asr-decoder.onnx_data",
    }
    for f in output_dir.iterdir():
        if f.suffix in (".data", ".onnx.data") and f.name not in keep:
            f.unlink()
        elif f.name.startswith("onnx__"):
            f.unlink()

    processor.save_pretrained(str(output_dir))

    print(f"[ASR] Export complete: {output_dir}")
    _list_files(output_dir)

# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Qwen3 models to ONNX format"
    )
    parser.add_argument("--asr", action="store_true", help="Export Qwen3-ASR-0.6B")
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Skip INT4 quantization (keep FP16 weights)",
    )
    parser.add_argument(
        "--fp16-decoder", action="store_true",
        help="Keep decoder as FP16 (skip INT4 quantization on decoder only). "
             "Encoder stays FP16 by default. Use for mixed-precision export "
             "where INT4 autoregressive decoding degrades quality.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: models/<model-name>-onnx)",
    )
    parser.add_argument(
        "--asr-model", default="Qwen/Qwen3-ASR-0.6B",
        help="HuggingFace model ID for ASR",
    )
    args = parser.parse_args()

    if not args.asr:
        parser.error("Specify --asr")

    do_quant = not args.no_quantize
    fp16_decoder = args.fp16_decoder

    if args.asr:
        out = args.output_dir or Path("models/qwen3-asr-0.6b-onnx")
        export_asr(out, args.asr_model, quantize=do_quant and not fp16_decoder)


if __name__ == "__main__":
    main()
