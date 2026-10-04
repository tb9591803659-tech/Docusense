import "./globals.css";
export const metadata = { title: "DocuSense AI", description: "Investigate documents. Follow the evidence." };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (<html lang="en"><body>{children}</body></html>);
}
