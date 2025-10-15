// express/server.js
const express = require("express");
const path = require("path");
const { Readable } = require("stream");

const API_BASE = process.env.API_BASE || "http://localhost:8000";
const app = express();

// Serve PDF.js viewer (cài pdfjs-dist)
const pdfjsRoot = path.join(path.dirname(require.resolve("pdfjs-dist/package.json")));
app.use("/pdfjs/web", express.static(path.join(pdfjsRoot, "web")));
app.use("/pdfjs/build", express.static(path.join(pdfjsRoot, "build")));

// Proxy file theo filename (giữ Range)
app.get("/files/:filename", async (req, res) => {
  const apiUrl = `${API_BASE}/fs/books/${encodeURIComponent(req.params.filename)}`;
  const headers = {};
  if (req.headers.range) headers["Range"] = req.headers.range;
  if (req.headers.cookie) headers["Cookie"] = req.headers.cookie;
  if (req.headers.authorization) headers["Authorization"] = req.headers.authorization;

  let r;
  try { r = await fetch(apiUrl, { headers }); }
  catch { res.status(502).send("Upstream unreachable"); return; }

  res.status(r.status);
  for (const [k, v] of r.headers.entries()) {
    if (["content-length","content-range","content-type","accept-ranges","content-disposition","cache-control","etag","last-modified"].includes(k.toLowerCase())) {
      res.setHeader(k, v);
    }
  }
  if (!r.body) { res.end(); return; }
  Readable.fromWeb(r.body).pipe(res);
});

// Link tiện đọc bằng viewer
app.get("/read/by-filename/:filename", (req, res) => {
  const fileUrl = `/files/${encodeURIComponent(req.params.filename)}`;
  const viewer = `/pdfjs/web/viewer.html?file=${encodeURIComponent(fileUrl)}#zoom=page-fit`;
  res.redirect(viewer);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Express on http://localhost:${PORT}`));
