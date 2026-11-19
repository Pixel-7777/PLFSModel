const state = {
  schema: null,
  numericFields: ["age", "years_formal_education", "tothrs_wrk", "days_worked"],
  categoricalFields: [],
};

const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});
const inrPrecise = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

const PERSON_FIELDS = [
  "age", "sex", "marst", "rel", "gedu_lvl", "years_formal_education",
  "tedu_lvl", "curr_att", "secondary", "voc", "voc_fld", "voc_typ",
];
const JOB_FIELDS = [
  "st", "dc", "sec", "ind_pas", "ocu_pas", "loc_pas", "etyp_pas",
  "wrkr_pas", "job_pas", "leave_pas", "ssec_pas", "has_sas", "dur_pas",
];
const ADVANCED_FIELDS = ["acws", "aind_cws", "ocu_cws"];

const $ = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", init);

async function init() {
  try {
    const response = await fetch("/api/schema");
    if (!response.ok) throw new Error("Could not load the form schema.");
    state.schema = await response.json();
  } catch (error) {
    showError(error.message);
    return;
  }

  state.categoricalFields = PERSON_FIELDS
    .concat(JOB_FIELDS)
    .concat(ADVANCED_FIELDS)
    .filter((field) => state.schema.fields[field]);

  buildForm();
  bindEvents();
  applyDefaults();
  updateFacts();
  await loadRecent();
}

function buildForm() {
  const person = $("person-fields");
  const job = $("job-fields");
  const advanced = $("advanced-fields");
  person.replaceChildren();
  job.replaceChildren();
  advanced.replaceChildren();

  addNumber(person, "age");
  addSelect(person, "sex");
  addSelect(person, "marst");
  addSelect(person, "rel");
  addSelect(person, "gedu_lvl");
  addNumber(person, "years_formal_education", { withUnknown: true });
  addNumber(person, "tothrs_wrk");
  addNumber(person, "days_worked");
  addSelect(person, "tedu_lvl");
  addSelect(person, "curr_att");
  addSelect(person, "secondary");
  addSelect(person, "voc");
  addSelect(person, "voc_fld");
  addSelect(person, "voc_typ");

  const derived = document.createElement("div");
  derived.className = "field wide";
  derived.innerHTML = `
    <label>Derived inputs</label>
    <div class="derived-row">
      <div>
        <input type="text" id="derived-experience" readonly aria-label="Derived potential experience">
        <span class="help">Potential experience = max(0, age − education − 6)</span>
      </div>
      <div>
        <input type="text" id="derived-avg-hours" readonly aria-label="Derived average daily hours">
        <span class="help">Average daily hours = total hours ÷ 7</span>
      </div>
    </div>
  `;
  person.appendChild(derived);

  addSelect(job, "st", { allowMissing: false });
  addSelect(job, "dc", { allowMissing: false });
  addSelect(job, "sec");
  addSelect(job, "ind_pas", { wide: true, hint: "type to search" });
  addSelect(job, "ocu_pas", { wide: true, hint: "type to search" });
  addSelect(job, "loc_pas");
  addSelect(job, "etyp_pas");
  addSelect(job, "wrkr_pas");
  addSelect(job, "job_pas");
  addSelect(job, "leave_pas");
  addSelect(job, "ssec_pas");
  addSelect(job, "has_sas");
  addSelect(job, "dur_pas");

  addSelect(advanced, "acws", { wide: true });
  addSelect(advanced, "aind_cws", { wide: true, hint: "type to search" });
  addSelect(advanced, "ocu_cws", { wide: true, hint: "type to search" });
}

function fieldWrapper(labelText, control, { wide = false, hint = "", help = "" } = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = "field" + (wide ? " wide" : "");

  const label = document.createElement("label");
  label.htmlFor = control.id;
  const name = document.createElement("span");
  name.textContent = labelText;
  label.appendChild(name);
  if (hint) {
    const hintEl = document.createElement("span");
    hintEl.className = "hint";
    hintEl.textContent = hint;
    label.appendChild(hintEl);
  }
  wrapper.appendChild(label);
  wrapper.appendChild(control);

  if (help) {
    const helpEl = document.createElement("span");
    helpEl.className = "help";
    helpEl.textContent = help;
    wrapper.appendChild(helpEl);
  }
  return wrapper;
}

function addNumber(container, field, { withUnknown = false } = {}) {
  const spec = state.schema.numeric_specs[field];
  const input = document.createElement("input");
  input.type = "number";
  input.id = `field-${field}`;
  input.name = field;
  input.dataset.field = field;
  input.min = spec.min;
  input.max = spec.max;
  input.step = spec.step;
  input.value = state.schema.numeric_defaults[field];
  const help = field === "age"
    ? "Training range: 11–86 years."
    : field === "tothrs_wrk"
      ? "Total hours across the seven-day reference week."
      : field === "days_worked"
        ? "Number of days with any hours worked (0–7)."
        : "Leave blank if unknown; the model uses the training median.";

  const wrapper = fieldWrapper(spec.label, input, { help });
  if (withUnknown) {
    const checkRow = document.createElement("div");
    checkRow.className = "checkbox-row";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = "edu-unknown";
    const label = document.createElement("label");
    label.htmlFor = "edu-unknown";
    label.textContent = "Not known (use model median)";
    checkRow.appendChild(checkbox);
    checkRow.appendChild(label);
    wrapper.appendChild(checkRow);
  }
  container.appendChild(wrapper);
}

function addSelect(container, field, { allowMissing = true, wide = false, hint = "" } = {}) {
  const spec = state.schema.fields[field];
  const select = document.createElement("select");
  select.id = `field-${field}`;
  select.name = field;
  select.dataset.field = field;

  if (allowMissing) {
    select.add(new Option(spec.missing_label || "Not specified (use model default)", "__missing__"));
  }
  for (const option of spec.options) {
    const opt = new Option(option.label, option.value);
    opt.title = option.label;
    select.add(opt);
  }

  const help = field === "ind_pas"
    ? "NIC-2008 five-digit industry. Type a code or keyword to jump."
    : field === "ocu_pas" || field === "ocu_cws"
      ? "NCO-2015 three-digit occupation group. Type a code or keyword to jump."
      : field === "acws"
        ? "The finalized model only saw regular-wage states 31, 71 and 72."
        : field === "st"
          ? "Select a state first; district options update automatically."
          : field === "dc"
            ? "Filtered to districts seen in the training data."
            : "";

  container.appendChild(fieldWrapper(spec.label, select, { wide, hint, help }));
}

function bindEvents() {
  $("wage-form").addEventListener("submit", onSubmit);
  $("reset-button").addEventListener("click", () => {
    applyDefaults();
    hideError();
    $("result").hidden = true;
    $("empty-state").hidden = false;
  });

  const age = $("field-age");
  const education = $("field-years_formal_education");
  const hours = $("field-tothrs_wrk");
  for (const input of [age, education, hours]) {
    input.addEventListener("input", updateDerived);
  }
  $("edu-unknown").addEventListener("change", () => {
    education.disabled = $("edu-unknown").checked;
    if (education.disabled) education.value = "";
    updateDerived();
  });

  $("field-st").addEventListener("change", () => {
    populateDistricts($("field-st").value, null);
  });
}

function applyDefaults() {
  const categorical = state.schema.defaults;
  const numeric = state.schema.numeric_defaults;

  for (const field of ["age", "years_formal_education", "tothrs_wrk", "days_worked"]) {
    const input = $(`field-${field}`);
    if (input) input.value = numeric[field];
  }
  $("edu-unknown").checked = false;
  $("field-years_formal_education").disabled = false;

  for (const field of state.categoricalFields) {
    if (field === "dc" || field === "state_district") continue;
    setSelectValue(field, categorical[field]);
  }

  const defaultState = categorical.st || (state.schema.fields.st.options[0]?.value);
  if (defaultState) {
    $("field-st").value = defaultState;
    populateDistricts(defaultState, categorical.dc);
  }

  updateDerived();
}

function setSelectValue(field, value) {
  const select = $(`field-${field}`);
  if (!select) return;
  if (value === null || value === undefined || value === "") {
    select.value = "__missing__";
    return;
  }
  const wanted = String(value);
  const exists = Array.from(select.options).some((option) => option.value === wanted);
  if (exists) {
    select.value = wanted;
  } else if (select.querySelector('option[value="__missing__"]')) {
    select.value = "__missing__";
  }
}

function populateDistricts(stateValue, selectedDistrict) {
  const select = $("field-dc");
  if (!select) return;
  select.replaceChildren();

  const districts = state.schema.districts_by_state[String(stateValue)] || [];
  if (!districts.length) {
    select.add(new Option("No district options", ""));
    select.disabled = true;
    return;
  }

  for (const district of districts) {
    const option = new Option(district.label, district.value);
    option.dataset.stateDistrict = district.state_district;
    select.add(option);
  }
  select.disabled = false;

  const wanted = selectedDistrict === null || selectedDistrict === undefined
    ? null
    : String(selectedDistrict);
  const found = districts.find((district) => district.value === wanted);
  select.value = found ? found.value : districts[0].value;
}

function updateDerived() {
  const age = numericValue("age");
  const educationUnknown = $("edu-unknown")?.checked;
  const education = educationUnknown ? null : numericValue("years_formal_education");

  const experience = age !== null && education !== null
    ? Math.max(0, age - education - 6)
    : null;
  const experienceEl = $("derived-experience");
  if (experienceEl) {
    experienceEl.value = experience === null ? "Imputed by model" : `${experience.toFixed(0)} years`;
  }

  const hours = numericValue("tothrs_wrk");
  const avgEl = $("derived-avg-hours");
  if (avgEl) {
    avgEl.value = hours === null ? "Imputed by model" : `${(hours / 7).toFixed(2)} hours`;
  }
}

function numericValue(field) {
  const input = $(`field-${field}`);
  if (!input) return null;
  const raw = input.value.trim();
  if (raw === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function payloadFromForm() {
  const inputs = {};

  const age = numericValue("age");
  if (age === null) throw new Error("Age is required.");
  if (age < 11 || age > 86) throw new Error("Age must be between 11 and 86 for this model.");
  inputs.age = age;

  const educationUnknown = $("edu-unknown").checked;
  const education = educationUnknown ? null : numericValue("years_formal_education");
  if (!educationUnknown && education === null) {
    throw new Error("Enter years of formal education, or tick “Not known”.");
  }
  if (education !== null && (education < 0 || education > 25)) {
    throw new Error("Years of formal education must be between 0 and 25.");
  }
  inputs.years_formal_education = education;

  const hours = numericValue("tothrs_wrk");
  if (hours === null) throw new Error("Total hours in the reference week is required.");
  if (hours < 0 || hours > 140) throw new Error("Total hours must be between 0 and 140.");
  inputs.tothrs_wrk = hours;

  const days = numericValue("days_worked");
  if (days === null) throw new Error("Days worked in the reference week is required.");
  if (days < 0 || days > 7) throw new Error("Days worked must be between 0 and 7.");
  inputs.days_worked = days;

  for (const field of state.categoricalFields) {
    const select = $(`field-${field}`);
    if (!select) continue;
    const value = select.value;
    inputs[field] = value === "__missing__" || value === "" ? null : value;
  }

  if (!inputs.st) throw new Error("State / Union Territory is required.");
  if (!inputs.dc) throw new Error("District is required.");

  return inputs;
}

async function onSubmit(event) {
  event.preventDefault();
  hideError();

  let inputs;
  try {
    inputs = payloadFromForm();
  } catch (error) {
    showError(error.message);
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    renderResult(data);
    await loadRecent();
  } catch (error) {
    showError(error.message || "Prediction failed.");
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  const button = $("submit-button");
  button.disabled = isLoading;
  button.textContent = isLoading ? "Estimating…" : "Estimate monthly wage";
}

function showError(message) {
  const box = $("form-error");
  box.textContent = message;
  box.hidden = false;
  box.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function hideError() {
  const box = $("form-error");
  box.hidden = true;
  box.textContent = "";
}

function renderResult(data) {
  const prediction = data.prediction;
  $("empty-state").hidden = true;
  $("result").hidden = false;

  $("median-value").textContent = inr.format(prediction.q50);
  $("range-value").textContent = `${inr.format(prediction.q05)} – ${inr.format(prediction.q95)}`;
  $("q05-value").textContent = inr.format(prediction.q05);
  $("q50-value").textContent = inr.format(prediction.q50);
  $("q95-value").textContent = inr.format(prediction.q95);
  $("width-value").textContent = inr.format(prediction.interval_width);

  const max = Math.max(prediction.q95 * 1.15, prediction.q50 * 2, 10000);
  const bandLeft = Math.max(0, Math.min(100, (prediction.q05 / max) * 100));
  const bandWidth = Math.max(1.5, Math.min(100 - bandLeft, ((prediction.q95 - prediction.q05) / max) * 100));
  const markerLeft = Math.max(0, Math.min(100, (prediction.q50 / max) * 100));

  $("interval-band").style.left = `${bandLeft}%`;
  $("interval-band").style.width = `${bandWidth}%`;
  $("interval-marker").style.left = `${markerLeft}%`;
  $("interval-max").textContent = inr.format(max);

  const warningBox = $("warning-box");
  const warningList = $("warnings");
  warningList.replaceChildren();
  if (data.warnings && data.warnings.length) {
    for (const warning of data.warnings) {
      const item = document.createElement("li");
      item.textContent = warning;
      warningList.appendChild(item);
    }
    warningBox.hidden = false;
  } else {
    warningBox.hidden = true;
  }

  if (window.matchMedia("(max-width: 1080px)").matches) {
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function loadRecent() {
  const body = $("recent-body");
  try {
    const response = await fetch("/api/predictions?limit=8");
    if (!response.ok) throw new Error("Could not load recent predictions.");
    const data = await response.json();
    body.replaceChildren();

    if (!data.items.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 4;
      cell.className = "empty-row";
      cell.textContent = "No predictions yet.";
      row.appendChild(cell);
      body.appendChild(row);
      return;
    }

    for (const item of data.items) {
      const row = document.createElement("tr");
      const summary = item.summary || {};
      const profile = [
        summary.age != null ? `${summary.age} yrs` : null,
        summary.sex,
        summary.state,
      ].filter(Boolean).join(" · ");

      const when = document.createElement("td");
      when.textContent = new Date(item.created_at).toLocaleString("en-IN", {
        dateStyle: "medium",
        timeStyle: "short",
      });

      const profileCell = document.createElement("td");
      profileCell.textContent = profile || "—";
      profileCell.title = [
        summary.district,
        summary.sector,
        summary.occupation,
        summary.industry,
      ].filter(Boolean).join(" · ");

      const median = document.createElement("td");
      median.className = "median-cell";
      median.textContent = inr.format(item.q50);

      const range = document.createElement("td");
      range.className = "range-cell";
      range.textContent = `${inr.format(item.q05)} – ${inr.format(item.q95)}`;

      row.append(when, profileCell, median, range);
      body.appendChild(row);
    }
  } catch (error) {
    body.replaceChildren();
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.className = "empty-row";
    cell.textContent = "Recent predictions are unavailable.";
    row.appendChild(cell);
    body.appendChild(row);
  }
}

function updateFacts() {
  const model = state.schema.model;
  const metrics = model.test_metrics;
  $("fact-coverage").textContent = `${(metrics.coverage_90 * 100).toFixed(1)}%`;
  $("fact-mae").textContent = inr.format(metrics.mae);
  $("fact-n").textContent = metrics.n.toLocaleString("en-IN");
  $("model-version").textContent = `${model.model_version} · weights ${model.weights_sha256.slice(0, 12)}…`;
}
