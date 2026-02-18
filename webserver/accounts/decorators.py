from functools import wraps
from django.shortcuts import redirect
from .models import Consent, Participant


def consent_required(view_func):
    """
    Decorator to check if user has given consent.
    Redirects to consent page if user hasn't consented.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            # If not authenticated, login_required should handle it
            return view_func(request, *args, **kwargs)
        
        # Check if user has consented
        try:
            participant = Participant.objects.get(user=request.user)
            if not participant.agreed_at or participant.withdrawn_at:
                return redirect(f'/accounts/consent/?next={request.path}')
        except Participant.DoesNotExist:
            participant = Participant(user=request.user)
            participant.save()
            return redirect(f'/accounts/consent/?next={request.path}')
        
        return view_func(request, *args, **kwargs)
    
    return _wrapped_view

