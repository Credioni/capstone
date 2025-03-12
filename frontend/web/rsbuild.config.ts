// import { defineConfig } from '@rsbuild/core';
// import { pluginReact } from '@rsbuild/plugin-react';

// export default defineConfig({
//   plugins: [pluginReact()],
// });

import { defineConfig } from '@rsbuild/core';
import { pluginReact } from '@rsbuild/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
  plugins: [pluginReact()],
  server: {
    // headers: {
    //   'Content-Type': 'application/json',  // Ensure correct MIME type
    // },
    //staticDir: path.join(__dirname, '../', 'lynxjs', 'dist'),
    publicDir: [
      {
        name: path.join(
          __dirname,
          '../',
          'lynxjs',
          'dist',
        ),
      },
    ],
  },
});