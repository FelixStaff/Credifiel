import { useState } from "react";
import HomePanel from "./components/HomePanel";
import ClusterPanel from "./components/charts/ClusterPanel";
import RevenueProjection from "./components/RevenueProjection";
import Sustento from "./components/charts/Sustento";
import Inversion from "./components/charts/Cobranza";
import OptimoVsReal from "./components/charts/OptimoVsReal";
import logo from "./assets/logo.png";
import {
  FaHome,
  FaProjectDiagram,
  FaChartLine,
  FaMoneyBillWave,
  FaChartPie,
  FaCogs,
} from "react-icons/fa";

export default function App() {
  const [section, setSection] = useState("home");

  const menuItemStyle = (key: string) =>
    `flex items-center space-x-3 cursor-pointer transition-colors duration-300 ${
      section === key ? "text-[#1D99D6]" : "text-white hover:text-[#1D99D6]"
    }`;

  return (
    <main className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-[16vw] sticky bg-[#1F2937] text-white shadow-md py-6 px-4 flex flex-col gap-6 border-r border-[#2c3e50]">
        {/* Logo */}
        <div className="flex items-center justify-center px-2 mb-2">
          <img
            src={logo}
            alt="Logo"
            className="h-10 object-contain drop-shadow-md"
          />
        </div>

        {/* General */}
        <div>
          <h3 className="text-xs uppercase text-gray-400 mb-2">General</h3>
          <ul className="space-y-3">
            <li className={menuItemStyle("home")} onClick={() => setSection("home")}>
              <FaHome className="text-lg" />
              <span>Home</span>
            </li>
                <li className={menuItemStyle("optimo")} onClick={() => setSection("optimo")}>
              <FaCogs className="text-lg" />
              <span>ROI</span>
            </li>
          </ul>
        </div>

        {/* Clustering */}
        <div>
          <h3 className="text-xs uppercase text-gray-400 mb-2">Clustering</h3>
          <ul className="space-y-3">
            <li className={menuItemStyle("cluster")} onClick={() => setSection("cluster")}>
              <FaProjectDiagram className="text-lg" />
              <span>Marker Clustering</span>
            </li>
          </ul>
        </div>

        {/* Módulos */}
        <div>
          <h3 className="text-xs uppercase text-gray-400 mb-2">Módulos</h3>
          <ul className="space-y-3">
            <li className={menuItemStyle("inversion")} onClick={() => setSection("inversion")}>
              <FaChartLine className="text-lg" />
              <span>Cobranza</span>
            </li>
          </ul>
        </div>

        {/* Fundamento Analítico */}
        <div>
          <h3 className="text-xs uppercase text-gray-400 mb-2">Fundamento Analítico</h3>
          <ul className="space-y-3">
            <li className={menuItemStyle("sustento")} onClick={() => setSection("sustento")}>
              <FaChartPie className="text-lg" />
              <span>Sustento</span>
            </li>
          </ul>
        </div>
      </aside>

      {/* Panel principal */}
      <section
        className={`flex-1 relative p-6 overflow-y-auto bg-gradient-to-br from-[#1F2937] via-[#2D3748] to-[#111827] text-white`}
      >
        {section === "home" && <HomePanel />}
        {section === "cluster" && <ClusterPanel />}
        {section === "projection" && <RevenueProjection />}
        {section === "sustento" && <Sustento />}
        {section === "inversion" && <Inversion />}
        {section === "optimo" && <OptimoVsReal />}
      </section>
    </main>
  );
}
