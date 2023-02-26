//import '@/styles/globals.css';
import './App.scss';
import { ClerkProvider } from '@clerk/clerk-react';
import { Dosis } from '@next/font/google';
import SSRProvider from 'react-bootstrap/SSRProvider';

const dosis = Dosis({ subsets: ['latin'] });

function App({ Component, pageProps }) {
  return (
    <ClerkProvider
      publishableKey={
        'pk_test_dGhhbmtmdWwtY29sdC0xNC5jbGVyay5hY2NvdW50cy5kZXYk'
      }
      appearance={{
        variables: {
          colorPrimary: '#7447D7',
          colorText: '#271254',
          fontFamily: dosis.style.fontFamily,
        },
      }}
    >
      <SSRProvider>
        <main className={dosis.className}>
          <Component {...pageProps} />
        </main>
      </SSRProvider>
    </ClerkProvider>
  );
}

export default App;
