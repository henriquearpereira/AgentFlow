import React from 'react';
import ReactDOM from 'react-dom/client';
import './App.css';
import App from './App.tsx';

// Get the root element
const root = ReactDOM.createRoot(
  document.getElementById('root')
);

// Render the App component
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Optional: Web Vitals reporting
// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
// reportWebVitals();