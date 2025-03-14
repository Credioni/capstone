import { root } from '@lynx-js/react'
import { App } from './App.jsx'
import { MemoryRouter, Routes, Route } from 'react-router';

//root.render(<App />) // Default

import { FrontPage } from './pages/FrontPage.tsx';
import { ResultPage } from './pages/ResultPage.tsx';

root.render(
    <MemoryRouter>
        <Routes>
            <Route path="/" element={<App Page={FrontPage}/>} />
            <Route path="/search">
                <Route path="/search" element={<App Page={ResultPage}/>} />
                <Route path="/search/:query" element={<App Page={ResultPage}/>} />
            </Route>
        </Routes>
    </MemoryRouter>,
);

if (import.meta.webpackHot) {
    import.meta.webpackHot.accept()
}
