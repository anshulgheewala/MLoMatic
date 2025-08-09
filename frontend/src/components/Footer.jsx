// import React from 'react';
// import { Box, Typography, Link, Chip, Divider } from '@mui/material';
// import FavoriteIcon from '@mui/icons-material/Favorite';
// import AdbIcon from '@mui/icons-material/Adb'; // A generic tech icon for version

// function Footer() {
//   return (
//     <Box sx={{ p: 4, mt: 'auto'}}>
//       <Divider sx={{ mb: 0.3, }} />
//       <Box
//         sx={{
//           display: 'flex',
//           flexDirection: 'column',
//           alignItems: 'center',
//           textAlign: 'center',
//           gap: 2,
//           bgcolor: 'whitesmoke'
//         }}
//       >
//         {/* Copyright Notice */}
//         <Typography variant="body2" color="text.secondary">
//           © {new Date().getFullYear()} AutoML Platform. All rights reserved.
//         </Typography>

//         {/* Credits Section */}
//         <Box
//           sx={{
//             display: 'flex',
//             alignItems: 'center',
//             gap: 0.5,
//             color: 'text.secondary',
//           }}
//         >
//           <Typography variant="body2">Made with</Typography>
//           <FavoriteIcon sx={{ fontSize: 16, color: 'red' }} />
//           <Typography variant="body2">by</Typography>
//           <Link href="#" underline="hover" color="inherit">
//             Anshul
//           </Link>
//           <Typography variant="body2">And</Typography>
//           <Link href="#" underline="hover" color="inherit">
//             Dhruvesh
//           </Link>
//         </Box>

//         {/* Version Chip */}
//         <Chip
//           icon={<AdbIcon />}
//           label="Version - v1.0.0"
//           variant="outlined"
//           size="small"
//         />
//       </Box>
//     </Box>
//   );
// }

// export default Footer;
import React from 'react';
import { Box, Typography, Link, Chip, Container } from '@mui/material';
import FavoriteIcon from '@mui/icons-material/Favorite';
import AdbIcon from '@mui/icons-material/Adb'; // A generic tech icon for version

function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        py: 4, // Vertical padding
        mt: 'auto', // Pushes footer to the bottom of the page
        backgroundColor: (theme) =>
          theme.palette.mode === 'light'
            ? theme.palette.grey[200]
            : theme.palette.grey[800],
        borderTop: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Container maxWidth="lg">
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            textAlign: 'center',
            gap: 2,
          }}
        >
          {/* Copyright Notice */}
          <Typography variant="body2" color="text.secondary">
            © {new Date().getFullYear()} MLoMatic. All rights reserved.
          </Typography>

          {/* Credits Section */}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              color: 'text.secondary',
            }}
          >
            <Typography variant="body2">Made with</Typography>
            <FavoriteIcon sx={{ fontSize: 16, color: 'red' }} />
            <Typography variant="body2">by</Typography>
            <Link href="https://github.com/anshulgheewala" underline="hover" color="inherit" target="_blank"
  rel="noopener noreferrer">
              Anshul
            </Link>
            <span variant="body2">and</span>
            <Link href="https://github.com/Dhruvesh304" underline="hover" color="inherit" target="_blank"
  rel="noopener noreferrer">
              Dhruvesh
            </Link>
          </Box>

          {/* Version Chip */}
          <Chip
            icon={<AdbIcon />}
            label="Version - v1.0.0"
            variant="outlined"
            size="small"
          />
        </Box>
      </Container>
    </Box>
  );
}

export default Footer;