export function coerceTrainingPayload(form) {
  const payload = { ...form };
  for (const key of ["max_steps", "max_seq_length", "lora_r", "gradient_accumulation_steps", "logging_steps", "save_steps"]) {
    if (payload[key] === "" || payload[key] == null) {
      delete payload[key];
    } else {
      payload[key] = Number(payload[key]);
    }
  }
  return payload;
}
