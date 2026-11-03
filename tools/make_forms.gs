/**
 * HiAQ survey form builder (Google Apps Script). Run by a team member in their own Google
 * account at script.google.com; nothing here runs automatically.
 *
 * 1. `ha data hiaq-variants` writes data/hiaq/survey_variants.json.
 * 2. Replace the placeholder object assigned to VARIANTS below with the FULL content of that
 *    file (it is a JSON object, which is also a valid JavaScript object literal; do not wrap it
 *    in quotes).
 * 3. Run createAllForms(). Before creating anything it checks the pasted content: 25 variants x
 *    6 items, 150 distinct scenario IDs, and the fingerprint EXPECTED_VARIANTS_SHA256 (so a
 *    truncated or edited paste stops the run). It then creates 25 forms ("HiAQ form 01".."25"),
 *    each with the Appendix D text and 6 scenario items; the 2 optional questions come on a
 *    second page, after the answers, so their language options cannot cue how people type.
 *    It does NOT collect emails and does NOT require sign-in (no identifiers, spec §15.3).
 *    Forms are created NOT accepting responses: opening them is the pilot/launch step.
 * 4. Each form writes one "MANIFEST_JSON {...}" line to the execution log. Copy the whole log
 *    to D:\humanalign-rag\data\hiaq\forms_creation_log.txt and run
 *    `ha data hiaq-forms-manifest` (it checks every form against survey_variants.json and
 *    writes data/hiaq/forms_manifest.json).
 * 5. Later (not now): paste the response URLs into tools/redirect.html; link each form to a
 *    response Sheet; export each as form_XX.csv into D:\humanalign-rag\data\hiaq\responses\
 *    for `ha data hiaq-ingest`.
 */

var VARIANTS = { "variants": [] };  // <- replace this object with the content of survey_variants.json
var EXPECTED_VARIANTS_SHA256 = '0b3bb2e7130df3035edcc72ec162b4d02e449d60f533552e1054a60f90858ec6';
var N_VARIANTS = 25, PER_FORM = 6;
var FIRST_VARIANT = 1;  // only if a run stopped part-way: set to the first form NOT yet created

function sha256Hex_(text) {
  var bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, text,
                                      Utilities.Charset.UTF_8);
  return bytes.map(function (b) { return ('0' + ((b + 256) % 256).toString(16)).slice(-2); })
              .join('');
}

function checkVariants_(cfg) {
  var v = cfg.variants || [];
  var ids = {};
  if (v.length !== N_VARIANTS) throw new Error('expected ' + N_VARIANTS + ' variants, got ' + v.length);
  v.forEach(function (x, i) {
    if (x.variant !== i + 1) throw new Error('variant numbers must be 1..' + N_VARIANTS);
    if (x.items.length !== PER_FORM) throw new Error('variant ' + x.variant + ' has ' + x.items.length + ' items');
    x.items.forEach(function (it) { ids[it.scenario_id] = true; });
  });
  if (Object.keys(ids).length !== N_VARIANTS * PER_FORM) throw new Error('scenario IDs are not 150 distinct');
  var got = sha256Hex_(JSON.stringify(cfg.variants));
  if (got !== EXPECTED_VARIANTS_SHA256) {
    throw new Error('pasted variants differ from survey_variants.json (sha256 ' + got + ')');
  }
}

function createAllForms() {
  var cfg = VARIANTS;
  checkVariants_(cfg);
  cfg.variants.forEach(function (v) {
    if (v.variant < FIRST_VARIANT) return;
    var num = ('0' + v.variant).slice(-2);
    var form = FormApp.create('HiAQ form ' + num);
    form.setTitle(cfg.title);
    form.setDescription(cfg.intro);
    form.setCollectEmail(false);
    try {
      form.setRequireLogin(false);  // Google Workspace accounts only; consumer forms need no login
    } catch (e) {
      Logger.log('form ' + num + ': setRequireLogin not applicable (' + e.message + ')');
    }
    form.setLimitOneResponsePerUser(false);  // would require sign-in
    form.setShowLinkToRespondAgain(false);
    form.setAllowResponseEdits(false);
    form.setProgressBar(false);
    v.items.forEach(function (item) {
      form.addTextItem()
        .setTitle('"' + item.scenario + '"')
        .setHelpText(cfg.item_help)
        .setRequired(false);
    });
    form.addPageBreakItem();  // optional questions only after the answers (no language cue)
    form.addCheckboxItem()
      .setTitle(cfg.lang_question)
      .setChoiceValues(cfg.lang_options)
      .setRequired(false);
    form.addMultipleChoiceItem()
      .setTitle(cfg.device_question)
      .setChoiceValues(cfg.device_options)
      .setRequired(false);
    form.setAcceptingResponses(false);  // created closed; opened only at pilot/launch
    var titles = form.getItems(FormApp.ItemType.TEXT).map(function (it) { return it.getTitle(); });
    Logger.log('MANIFEST_JSON ' + JSON.stringify({
      form_number: v.variant,
      form_id: form.getId(),
      edit_url: form.getEditUrl(),
      response_url: form.getPublishedUrl(),
      scenario_ids: v.items.map(function (it) { return it.scenario_id; }),
      item_titles: titles,
      accepting_responses: form.isAcceptingResponses(),
      created_at: new Date().toISOString()
    }));
  });
}
