// import { defineConfig } from "vite";
// import react from "@vitejs/plugin-react";
// import tsconfigPaths from "vite-tsconfig-paths";
// import tagger from "@dhiwise/component-tagger";

// // https://vitejs.dev/config/
// export default defineConfig({
//   // This changes the out put dir from dist to build
//   // comment this out if that isn't relevant for your project
//   build: {
//     outDir: "build",
//     chunkSizeWarningLimit: 2000,
//   },
//   plugins: [tsconfigPaths(), react(), tagger()],
//   server: {
//     port: "4028",
//     host: "0.0.0.0",
//     strictPort: true,
//     allowedHosts: ['.amazonaws.com']
//   }
// });
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";
import tagger from "@dhiwise/component-tagger";

// https://vitejs.dev/config/
export default defineConfig({
  build: {
    // Changes output dir from dist to build
    outDir: "build",
    // Increased limit to suppress warnings, but manualChunks below fixes the root cause
    chunkSizeWarningLimit: 2000, 
    rollupOptions: {
      output: {
        manualChunks(id) {
          // 1. Split React and Core libs into a separate file
          if (id.includes('node_modules/react') || 
              id.includes('node_modules/react-dom') || 
              id.includes('node_modules/react-router')) {
            return 'vendor-react';
          }

          // 2. Split DhiWise/UI libraries (since you are using tagger)
          if (id.includes('node_modules/@dhiwise') || 
              id.includes('node_modules/@mui') || 
              id.includes('node_modules/antd')) {
            return 'vendor-ui';
          }

          // 3. Split heavy Data/Chart libraries (Common in Streamlit apps)
          if (id.includes('node_modules/plotly') || 
              id.includes('node_modules/apexcharts') ||
              id.includes('node_modules/recharts')) {
            return 'vendor-charts';
          }

          // 4. Catch-all: Put all other node_modules in a general vendor file
          if (id.includes('node_modules')) {
            return 'vendor-utils';
          }
        },
      },
    },
  },
  plugins: [tsconfigPaths(), react(), tagger()],
  server: {
    port: "4028",
    host: "0.0.0.0",
    strictPort: true,
    allowedHosts: ['.amazonaws.com']
  }
});