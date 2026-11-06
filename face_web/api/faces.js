// Vercel serverless function. Stores face descriptors (numbers, not photos) in Upstash Redis.
const URL_ = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL;
const TOKEN = process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN;

async function redis(cmd) {
  const r = await fetch(URL_, {
    method: "POST",
    headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" },
    body: JSON.stringify(cmd),
  });
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j.result;
}

module.exports = async (req, res) => {
  res.setHeader("Cache-Control", "no-store");
  if (!URL_ || !TOKEN) return res.status(200).json({ configured: false, people: {} });

  try {
    if (req.method === "GET") {
      const flat = (await redis(["HGETALL", "faces"])) || [];
      const people = {};
      for (let i = 0; i < flat.length; i += 2) people[flat[i]] = JSON.parse(flat[i + 1]);
      return res.status(200).json({ configured: true, people });
    }

    // Optional PIN so strangers with the link can't add/delete people
    const pin = process.env.ADMIN_PIN;
    if (pin && req.headers["x-pin"] !== pin) return res.status(401).json({ error: "PIN required" });

    if (req.method === "POST") {
      const { name, descriptors } = req.body || {};
      if (!name || !Array.isArray(descriptors) || !descriptors.length)
        return res.status(400).json({ error: "name and descriptors required" });
      await redis(["HSET", "faces", String(name).slice(0, 60), JSON.stringify(descriptors)]);
      return res.status(200).json({ ok: true });
    }
    if (req.method === "DELETE") {
      const name = req.query.name;
      if (!name) return res.status(400).json({ error: "name required" });
      await redis(["HDEL", "faces", name]);
      return res.status(200).json({ ok: true });
    }
    return res.status(405).json({ error: "method not allowed" });
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
};
