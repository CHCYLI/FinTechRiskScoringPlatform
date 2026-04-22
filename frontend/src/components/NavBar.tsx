import { NavLink } from "react-router-dom";
import { NAV_ITEMS } from "../lib/constants";

export default function NavBar() {
  return (
    <header className="navbar">
      <div className="navbar__brand">Risk Scoring Platform</div>
      <nav className="navbar__nav">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              isActive ? "nav-link nav-link--active" : "nav-link"
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
