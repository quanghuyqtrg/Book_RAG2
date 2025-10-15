const express = require('express');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 4000;

// CORS configuration for PDF.js compatibility
const corsOptions = {
  origin: '*', // Allow all origins (adjust for production)
  methods: ['GET', 'HEAD', 'OPTIONS'],
  allowedHeaders: ['Origin', 'X-Requested-With', 'Content-Type', 'Accept', 'Range'],
  credentials: false
};

app.use(cors(corsOptions));

// Additional headers for PDF streaming
app.use((req, res, next) => {
  res.header('Accept-Ranges', 'bytes');
  res.header('Cache-Control', 'public, max-age=31536000');
  next();
});

// Serve static files from the 'pdfs' folder
app.use('/pdfs', express.static(path.join(__dirname, 'books'), {
  setHeaders: (res, path, stat) => {
    if (path.endsWith('.pdf')) {
      res.setHeader('Content-Type', 'application/pdf');
      res.setHeader('Content-Disposition', 'inline'); // Display in browser, not download
    }
  }
}));

// Root endpoint to list all PDFs
app.get('/', (req, res) => {
  const pdfsDir = path.join(__dirname, 'books');
  
  // Check if pdfs directory exists
  if (!fs.existsSync(pdfsDir)) {
    fs.mkdirSync(pdfsDir, { recursive: true });
    return res.json({ 
      message: 'PDFs directory created. Please add PDF files to the /pdfs folder.',
      pdfs: []
    });
  }

  // Read all PDF files from the directory
  fs.readdir(pdfsDir, (err, files) => {
    if (err) {
      return res.status(500).json({ error: 'Unable to read PDFs directory' });
    }

    const pdfFiles = files.filter(file => file.toLowerCase().endsWith('.pdf'));
    
    res.json({
      message: 'PDF Server is running',
      totalPDFs: pdfFiles.length,
      pdfs: pdfFiles.map(file => ({
        name: file,
        url: `http://localhost:${PORT}/books/${file}`,
        viewerUrl: `http://localhost:${PORT}/viewer/${file}`
      }))
    });
  });
});

// Direct PDF access endpoint
app.get('/pdf/:filename', (req, res) => {
  const filename = req.params.filename;
  const filePath = path.join(__dirname, 'books', filename);
  
  // Security check - ensure filename doesn't contain path traversal
  if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
    return res.status(400).json({ error: 'Invalid filename' });
  }
  
  // Check if file exists
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'PDF not found' });
  }
  
  // Set proper headers for PDF
  res.setHeader('Content-Type', 'application/pdf');
  res.setHeader('Content-Disposition', 'inline');
  
  // Stream the file
  const stream = fs.createReadStream(filePath);
  stream.pipe(res);
  
  stream.on('error', (err) => {
    console.error('Error streaming PDF:', err);
    res.status(500).json({ error: 'Error reading PDF file' });
  });
});

// PDF viewer endpoint (for direct browser viewing)
app.get('/viewer/:filename', (req, res) => {
  const filename = req.params.filename;
  const pdfUrl = `http://localhost:${PORT}/pdf/${filename}`;
  
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Viewer - ${filename}</title>
        <style>
            body { margin: 0; font-family: Arial, sans-serif; }
            iframe { border: none; }
            .header { background: #333; color: white; padding: 10px; text-align: center; }
        </style>
    </head>
    <body>
        <div class="header">
            <h3>PDF Viewer: ${filename}</h3>
            <a href="${pdfUrl}" target="_blank" style="color: #4CAF50;">Open in new tab</a> | 
            <a href="${pdfUrl}" download style="color: #4CAF50;">Download</a>
        </div>
        <iframe src="${pdfUrl}" width="100%" height="90%"></iframe>
    </body>
    </html>
  `);
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    timestamp: new Date().toISOString(),
    port: PORT 
  });
});

// Handle 404 for all other routes
app.use('*', (req, res) => {
  res.status(404).json({ 
    error: 'Route not found',
    availableEndpoints: [
      'GET / - List all books',
      'GET /pdfs/:filename - Direct PDF access',
      'GET /pdf/:filename - PDF with proper headers',
      'GET /viewer/:filename - PDF viewer page',
      'GET /health - Health check'
    ]
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 PDF Server running on http://localhost:${PORT}`);
  console.log(`📁 Serving PDFs from: ${path.join(__dirname, 'pdfs')}`);
  console.log(`🌐 Access from network: http://192.168.10.4:${PORT}`);
  console.log(`\n📋 Available endpoints:`);
  console.log(`   • http://localhost:${PORT}/ - List all PDFs`);
  console.log(`   • http://localhost:${PORT}/pdfs/filename.pdf - Direct PDF access`);
  console.log(`   • http://localhost:${PORT}/pdf/filename.pdf - PDF with headers`);
  console.log(`   • http://localhost:${PORT}/viewer/filename.pdf - PDF viewer`);
});