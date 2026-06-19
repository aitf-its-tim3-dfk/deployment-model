import { readFileSync, writeFileSync, mkdirSync } from 'fs'
import { parse } from 'yaml'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const __dir = dirname(fileURLToPath(import.meta.url))

const yaml = readFileSync(join(__dir, 'openapi.yaml'), 'utf8')
const spec = parse(yaml)
const specJson = JSON.stringify(spec)

const html = `<!doctype html>
<html>
<head>
  <title>DFK Model API Reference</title>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
  <div id="app"></div>
  <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  <script>
    Scalar.createApiReference('#app', {
      content: ${specJson},
      defaultHttpClient: {
        targetKey: 'shell',
        clientKey: 'curl',
      },
      hideClientButton: false,
    })
  </script>
</body>
</html>`

mkdirSync(join(__dir, 'dist'), { recursive: true })
writeFileSync(join(__dir, 'dist', 'index.html'), html, 'utf8')
console.log('Built docs/dist/index.html')
