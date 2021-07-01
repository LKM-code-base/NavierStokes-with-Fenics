def set_coriolis_force(self):
        
        assert isinstance(Omega, dlfn.Constant)
        
        if self._mesh.geometry().dim() is 2:
            
            assert len(Omega) == 1
        
            set._coriolis_force = 2 * dlfn.as_vector((-Omega * sol_v[1], Omega * sol_v[0]))  
        
        else:
            
            assert len(Omega) == 3
            
            self._coriolis_force = 2 * dlfn.cross(Omega, sol_v)
            
def set_euler_force(self):
        
        assert isinstance(Alpha, dlfn.Constant):
        
        if self._mesh.geometry().dim() is 2:
            
            assert len(Alpha) == 1
        
            set._euler_force = dlfn.as_vector((-Alpha * sol_v[1], Alpha * sol_v[0]))  
        
        else:
            
            assert len(Alpha) == 3
            
            self._euler_force = dlfn.cross(Alpha, sol_v)
