//import './App.css';
import '@lynx-js/web-core/index.css';
import '@lynx-js/web-elements/index.css';
import '@lynx-js/web-core';

const App = () => {
  return (
    <div style={{height: '100vh' }}>
      <lynx-view
        style={{ height: '100%', width: '100%' }}
        url="/main.web.bundle"
      ></lynx-view>
    </div>
  );
};

export default App;

// import './App.css';

// const App = () => {
//   return (
//     <div className="content">
//       <h1>Rsbuild with React</h1>
//       <p>Start building amazing things with Rsbuild.</p>
//     </div>
//   );
// };

// export default App;
