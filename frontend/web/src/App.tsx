import './App.css';
import '@lynx-js/web-core/index.css';
import '@lynx-js/web-elements/index.css';
import '@lynx-js/web-core';



const App = () => {
  // const handleButtonClick = () => {
  //     const customEvent = new CustomEvent('customEvent', {
  //       detail: { message: 'Hello from Web!' },
  //     });
  //     window.dispatchEvent(customEvent);
  // };

  return (
    <div style={{ height: '100vh' }}>
      <lynx-view
        style={{ height: '100%', width: '100%' }}
        url="/main.web.bundle"
      ></lynx-view>
    </div>
  );
};

export default App;
