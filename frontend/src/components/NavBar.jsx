import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Divider,
} from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'; // A fitting icon for an ML app
import MenuIcon from '@mui/icons-material/Menu';

const navItems = ['Home', 'About', 'History'];

function NavBar() {
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleDrawerToggle = () => {
    setMobileOpen((prevState) => !prevState);
  };

  // This is the drawer (menu) that appears on mobile screens
  const drawer = (
    <Box onClick={handleDrawerToggle} sx={{ textAlign: 'center' }}>
      <Typography variant="h6" sx={{ my: 2 }}>
        AutoML
      </Typography>
      <Divider />
      <List>
        {navItems.map((item) => (
          <ListItem key={item} disablePadding>
            <ListItemButton sx={{ textAlign: 'center' }}>
              <ListItemText primary={item} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        component="nav"
        sx={{
          // Glassmorphism effect
          backgroundColor: 'rgba(255, 255, 255, 0.7)',
          backdropFilter: 'blur(10px)',
          boxShadow: 'none',
          borderBottom: '1px solid rgba(0, 0, 0, 0.12)',
        }}
      >
        <Toolbar>
          {/* Mobile Menu Icon */}
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' }, color: 'text.primary' }}
          >
            <MenuIcon />
          </IconButton>

          {/* Logo and Title */}
          <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
            <AutoAwesomeIcon sx={{ mr: 1, color: 'primary.main' }} />
            <Typography
              variant="h6"
              component="div"
              sx={{ color: 'text.primary', fontWeight: 'bold' }}
            >
              {/* AutoML Platform */}
              MLoMatic
            </Typography>
          </Box>

          {/* Desktop Navigation Links */}
          <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
            {navItems.map((item) => (
              <Button key={item} sx={{ color: 'text.primary' }}>
                {item}
              </Button>
            ))}
          </Box>
        </Toolbar>
      </AppBar>

      {/* Mobile Drawer */}
      <nav>
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: 240 },
          }}
        >
          {drawer}
        </Drawer>
      </nav>
      
      {/* This Box is a spacer to push your main content below the navbar */}
      <Box component="main" sx={{ p: 3 }}>
        <Toolbar /> 
      </Box>
    </Box>
  );
}

export default NavBar;